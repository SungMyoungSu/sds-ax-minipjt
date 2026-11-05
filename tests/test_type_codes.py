from type_codes import load_type_codes


def test_table_loads_with_source_and_status():
    t = load_type_codes()
    assert "1365" in t.source
    assert t.verified is True and t.as_of == "2026-11-05"  # 7단계에서 실제 응답으로 확인
    assert len(t.codes) == 15


def test_lookup():
    t = load_type_codes()
    assert t.name("0800") == "환경ㆍ생태계보호"
    assert t.code_for_name("환경ㆍ생태계보호") == "0800"
    assert t.code_for_name("환경·생태계보호") == "0800"  # 가운뎃점 차이 무시
    assert t.code_for_name("농어촌봉사") == "0600"  # 공백 차이 무시
    assert t.name("9900") is None and t.name("1500") == "기타"
    assert t.is_valid("0100")
    assert not t.is_valid("ZZZZ")
    assert t.name("ZZZZ") is None


def test_codes_are_unique_four_digit_strings():
    t = load_type_codes()
    assert all(len(c) == 4 and c.isdigit() for c in t.codes)
    assert len(set(t.codes.values())) == len(t.codes)
