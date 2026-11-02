"""Import medicine disease labels and sources from an .xlsx workbook.

The importer uses the Python standard library so it works without pandas or
openpyxl. It expects columns A-C to contain disease, medicine name, and source.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree


NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)


def normalize(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "", ascii_value)


def read_workbook(path: Path) -> list[tuple[str, str, str]]:
    ns = {"m": NAMESPACE}
    with ZipFile(path) as workbook:
        workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        relationship_map = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_xml = ElementTree.fromstring(
                workbook.read("xl/sharedStrings.xml")
            )
            shared_strings = [
                "".join(text.text or "" for text in item.iter(f"{{{NAMESPACE}}}t"))
                for item in shared_xml.findall("m:si", ns)
            ]

        sheet = workbook_xml.find("m:sheets/m:sheet", ns)
        if sheet is None:
            raise ValueError("The workbook has no worksheet.")
        relationship_id = sheet.attrib[f"{{{RELATIONSHIP_NAMESPACE}}}id"]
        target = relationship_map[relationship_id].lstrip("/")
        worksheet_path = target if target.startswith("xl/") else f"xl/{target}"
        worksheet = ElementTree.fromstring(workbook.read(worksheet_path))

        rows: list[tuple[str, str, str]] = []
        for row in worksheet.findall(".//m:sheetData/m:row", ns):
            values = []
            for cell in row.findall("m:c", ns):
                value = cell.find("m:v", ns)
                if value is None:
                    values.append("")
                elif cell.attrib.get("t") == "s":
                    values.append(shared_strings[int(value.text)])
                else:
                    values.append(value.text or "")
            if len(values) >= 3 and values[1].strip() != "Tên Thuốc":
                if values[1].strip():
                    rows.append(
                        (values[0].strip(), values[1].strip(), values[2].strip())
                    )
        return rows


def resolve_medicine_id(
    name: str,
    medicine_names: dict[str, str],
) -> str | None:
    aliases = {
        "zinc": "Vien_DiUngDa,DauRang_Zinc_009",
        "newartisonicdanapha": "Vien_DiUngDa_NewArtisonicDanaph_007",
        "tiphapredm": "Vien_DauRang,DiUngDa_TiprapredM_027",
        "berberindomesco": "Vien_TieuChay_Berberin_036",
    }
    key = normalize(name)
    if key in aliases:
        return aliases[key]

    matches = [
        medicine_id
        for medicine_id, medicine_name in medicine_names.items()
        if key == normalize(medicine_name)
        or key in normalize(medicine_name)
        or normalize(medicine_name) in key
    ]
    return matches[0] if len(matches) == 1 else None


def import_knowledge(
    workbook_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> tuple[int, list[str]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    medicine_names = {
        item["medicine_id"]: item["medicine_name"] for item in metadata
    }
    entries = {
        medicine_id: {
            "medicine_id": medicine_id,
            "medicine_name": medicine_name,
            "active_ingredient": None,
            "strength": None,
            "indication": None,
            "warning": None,
            "source_document_id": None,
            "source_url": None,
            "source_note": None,
        }
        for medicine_id, medicine_name in medicine_names.items()
    }
    unmatched: list[str] = []
    for disease, medicine_name, source in read_workbook(workbook_path):
        medicine_id = resolve_medicine_id(medicine_name, medicine_names)
        if medicine_id is None:
            unmatched.append(medicine_name)
            continue
        entry = entries[medicine_id]
        entry["indication"] = disease
        if source.startswith(("http://", "https://")):
            entry["source_url"] = source
        elif source:
            entry["source_note"] = source

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(list(entries.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(entries) - len(unmatched), unmatched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import medicine sources from an Excel workbook."
    )
    parser.add_argument("workbook", type=Path)
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/pill_index/pills.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/medicine_knowledge/medicines.json"),
    )
    args = parser.parse_args()
    imported, unmatched = import_knowledge(
        args.workbook, args.metadata, args.output
    )
    print(f"Imported knowledge for {imported} medicine entries.")
    if unmatched:
        print(f"Unmatched workbook names: {', '.join(unmatched)}")


if __name__ == "__main__":
    main()
