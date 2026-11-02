"""Xây đồ thị Neo4j (quan hệ HO_TRO và XUNG_DOT) từ 2 file CSV và đọc dữ liệu ngược ra."""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import pandas as pd
from neo4j import GraphDatabase

from config import Config

# Tên cột bắt buộc trong từng file CSV
COT_HO_TRO = ("Tên thuốc", "Thuốc hỗ trợ", "Cách thức hỗ trợ cụ thể")
COT_XUNG_DOT = ("Thuốc chính", "Thuốc xung đột", "Lý do xung đột")


def _doc_csv(path, cot_bat_buoc) -> pd.DataFrame:
    df = pd.read_csv(path)
    thieu = [c for c in cot_bat_buoc if c not in df.columns]
    if thieu:
        raise ValueError(f"File {path} thiếu cột: {thieu}. Các cột hiện có: {list(df.columns)}")
    return df


def _xoa_du_lieu_cu(session) -> None:
    # CẢNH BÁO: xoá TOÀN BỘ node :Thuoc trong database đang kết nối.
    session.run("MATCH (n:Thuoc) DETACH DELETE n")


def _create_indexes(session) -> None:
    session.run("CREATE INDEX thuoc_ten IF NOT EXISTS FOR (t:Thuoc) ON (t.ten)")


def _import_ho_tro(session, df: pd.DataFrame) -> None:
    rows, seen = [], set()
    for _, row in df.iterrows():
        a, b = str(row["Tên thuốc"]).strip(), str(row["Thuốc hỗ trợ"]).strip()
        if (a, b) not in seen and (b, a) not in seen:
            seen.add((a, b))
            rows.append({"ten_thuoc": a, "thuoc_ho_tro": b,
                         "cach_thuc": str(row["Cách thức hỗ trợ cụ thể"]).strip()})
    session.run("""
        UNWIND $rows AS row
        MERGE (a:Thuoc {ten: row.ten_thuoc})
        MERGE (b:Thuoc {ten: row.thuoc_ho_tro})
        MERGE (a)-[r:HO_TRO]->(b)
          ON CREATE SET r.cach_thuc = row.cach_thuc
          ON MATCH  SET r.cach_thuc = row.cach_thuc
    """, rows=rows)


def _import_xung_dot(session, df: pd.DataFrame) -> None:
    rows, seen = [], set()
    for _, row in df.iterrows():
        a, b = str(row["Thuốc chính"]).strip(), str(row["Thuốc xung đột"]).strip()
        if (a, b) not in seen and (b, a) not in seen:
            seen.add((a, b))
            rows.append({"thuoc_chinh": a, "thuoc_xung_dot": b,
                         "ly_do": str(row["Lý do xung đột"]).strip()})
    session.run("""
        UNWIND $rows AS row
        MERGE (a:Thuoc {ten: row.thuoc_chinh})
        MERGE (b:Thuoc {ten: row.thuoc_xung_dot})
        MERGE (a)-[r:XUNG_DOT]->(b)
          ON CREATE SET r.ly_do = row.ly_do
          ON MATCH  SET r.ly_do = row.ly_do
    """, rows=rows)


def _lay_triple(session) -> List[Tuple[str, str, str]]:
    rows_ht = session.run("MATCH (a:Thuoc)-[:HO_TRO]->(b:Thuoc) RETURN a.ten AS h, 'HO_TRO' AS r, b.ten AS t").data()
    rows_xd = session.run("MATCH (a:Thuoc)-[:XUNG_DOT]->(b:Thuoc) RETURN a.ten AS h, 'XUNG_DOT' AS r, b.ten AS t").data()
    return [(d["h"], d["r"], d["t"]) for d in rows_ht + rows_xd]


def _lay_xung_dot(session) -> Dict[FrozenSet[str], str]:
    rows = session.run("MATCH (a:Thuoc)-[r:XUNG_DOT]->(b:Thuoc) RETURN a.ten AS a, b.ten AS b, r.ly_do AS ly_do").data()
    return {frozenset({r["a"], r["b"]}): r["ly_do"] for r in rows}


def _lay_ho_tro(session) -> Dict[FrozenSet[str], str]:
    rows = session.run("MATCH (a:Thuoc)-[r:HO_TRO]->(b:Thuoc) RETURN a.ten AS a, b.ten AS b, r.cach_thuc AS cach_thuc").data()
    return {frozenset({r["a"], r["b"]}): r["cach_thuc"] for r in rows}


def xay_do_thi_va_lay_du_lieu(cfg: Config):
    """Nạp 2 file CSV vào Neo4j rồi đọc lại.

    Trả về (cap_xung_dot, cap_ho_tro, all_triples):
      cap_xung_dot / cap_ho_tro : {frozenset({thuoc_a, thuoc_b}): ghi chú}
      all_triples               : [(head, relation, tail), ...] để train PyKEEN
    """
    print("=" * 60)
    print("GIAI ĐOẠN 1: XÂY ĐỒ THỊ HỖ TRỢ + XUNG ĐỘT TRONG NEO4J")
    print("=" * 60)

    df_ho_tro = _doc_csv(cfg.file_ho_tro, COT_HO_TRO)
    df_xung_dot = _doc_csv(cfg.file_xung_dot, COT_XUNG_DOT)

    with GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)) as driver:
        with driver.session() as session:
            _xoa_du_lieu_cu(session)
            _create_indexes(session)
            _import_ho_tro(session, df_ho_tro)
            _import_xung_dot(session, df_xung_dot)

            cap_xung_dot = _lay_xung_dot(session)
            cap_ho_tro = _lay_ho_tro(session)
            all_triples = _lay_triple(session)

    print(f"  Đã nạp {len(cap_ho_tro)} cặp hỗ trợ, {len(cap_xung_dot)} cặp xung đột.")
    print(f"  Tổng {len(all_triples)} triple để train PyKEEN.\n")
    return cap_xung_dot, cap_ho_tro, all_triples
