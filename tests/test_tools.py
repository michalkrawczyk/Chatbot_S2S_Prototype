"""Tests for tools/ modules: datasheet_manager, file_manager, tool_register."""
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 4. test_datasheet_manager_csv_pipeline
# ---------------------------------------------------------------------------

def test_datasheet_manager_csv_pipeline(tmp_path):
    from tools.datasheet_manager import DatasheetManager

    dm = DatasheetManager()

    # 1. get_description() before load → dict with "error" key.
    assert "error" in dm.get_description()

    # 2. df_as_str() before load → sentinel string.
    assert "No data loaded" in dm.df_as_str()

    csv_path = tmp_path / "test.csv"
    csv_path.write_text("name,value\nalice,10\nbob,20\n")

    # 3. load_csv(path) → no exception; df_filepath == path.
    dm.load_csv(str(csv_path))
    assert dm.df_filepath == str(csv_path)

    # 4. get_description() after load → contains "rows", "columns", "column_names".
    desc = dm.get_description()
    assert "rows" in desc
    assert "columns" in desc
    assert "column_names" in desc

    # 5. get_chunk(rows=0) → single-row DataFrame.
    chunk = dm.get_chunk(rows=0)
    assert len(chunk) == 1

    # 6. get_chunk(columns="name") → DataFrame with only that column.
    chunk = dm.get_chunk(columns="name")
    assert list(chunk.columns) == ["name"]

    # 7. calculate_statistics on numeric col → dict with requested stat keys.
    stats = dm.calculate_statistics(columns="value", stats=["mean", "min", "max"])
    assert "value" in stats
    assert "mean" in stats["value"]
    assert "min" in stats["value"]
    assert "max" in stats["value"]

    # 8. calculate_statistics on string col → dict with "error" key.
    result = dm.calculate_statistics(columns="name")
    assert "error" in result

    # 9. df_as_str(limit_length=10) → string of length ≤ 10.
    assert len(dm.df_as_str(limit_length=10)) <= 10


# ---------------------------------------------------------------------------
# 5. test_datasheet_manager_excel_pipeline
# ---------------------------------------------------------------------------

def test_datasheet_manager_excel_pipeline(tmp_path):
    from tools.datasheet_manager import DatasheetManager

    dm = DatasheetManager()

    excel_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(str(excel_path)) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"b": [3, 4]}).to_excel(writer, sheet_name="Sheet2", index=False)

    # 1. get_sheet_names → list with both sheet names.
    names = dm.get_sheet_names(str(excel_path))
    assert "Sheet1" in names
    assert "Sheet2" in names

    # 2. load_excel(path, sheet_name=0) → loads first sheet; df_filepath == path.
    dm.load_excel(str(excel_path), sheet_name=0)
    assert dm.df_filepath == str(excel_path)

    # 3. load_excel by name → no exception.
    dm.load_excel(str(excel_path), sheet_name="Sheet2")

    # 4. load_excel with non-existent sheet name → raises ValueError.
    with pytest.raises(ValueError):
        dm.load_excel(str(excel_path), sheet_name="NonExistent")

    # 5. load_excel with out-of-range index → raises IndexError.
    with pytest.raises(IndexError):
        dm.load_excel(str(excel_path), sheet_name=99)


# ---------------------------------------------------------------------------
# 6. test_filesystem_manager_index_and_read_pipeline
# ---------------------------------------------------------------------------

def test_filesystem_manager_index_and_read_pipeline(tmp_path, monkeypatch):
    import tools.file_manager as fm_module

    monkeypatch.setattr(fm_module, "DATA_FILES_DIR", str(tmp_path))

    from tools.file_manager import FileSystemManager

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    # 1. Construct → index.json and global_file_helper.json created in DATA_FILES_DIR.
    fsm = FileSystemManager(memory_dir=str(memory_dir))
    assert (tmp_path / "index.json").exists()
    assert (tmp_path / "global_file_helper.json").exists()

    # 2. Write .txt into memory_dir; list_files() contains it.
    txt = memory_dir / "sample.txt"
    txt.write_text("hello world")
    files = fsm.list_files()
    assert any("sample.txt" in f for f in files)

    # 3. read_file_safe with filename only → (True, content).
    ok, content = fsm.read_file_safe("sample.txt")
    assert ok
    assert "hello world" in content

    # 4. Write .csv; read_file_safe → (True, non-empty string).
    csv = memory_dir / "data.csv"
    csv.write_text("col1,col2\n1,2\n")
    ok, content = fsm.read_file_safe("data.csv")
    assert ok
    assert content

    # 5. read_file_safe on non-existent file → (False, error_msg).
    ok, msg = fsm.read_file_safe("nonexistent.txt")
    assert not ok
    assert msg

    # 6. read_file_safe on unsupported extension → (False, error_msg).
    exe = memory_dir / "test.exe"
    exe.write_text("binary")
    ok, msg = fsm.read_file_safe("test.exe")
    assert not ok
    assert msg

    # 7. read_file (legacy) returns same content as read_file_safe for .txt.
    legacy = fsm.read_file("sample.txt")
    _, safe = fsm.read_file_safe("sample.txt")
    assert legacy == safe


# ---------------------------------------------------------------------------
# 7. test_filesystem_manager_global_helper_pipeline
# ---------------------------------------------------------------------------

def test_filesystem_manager_global_helper_pipeline(tmp_path, monkeypatch):
    import tools.file_manager as fm_module

    monkeypatch.setattr(fm_module, "DATA_FILES_DIR", str(tmp_path))

    from tools.file_manager import FileSystemManager

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    fsm = FileSystemManager(memory_dir=str(memory_dir))

    # 1. Empty manager → sentinel string.
    assert fsm.get_global_helper_as_context() == "No files in global helper catalog."

    # 2. add_file_to_global_helper on existing file → no exception.
    notes = memory_dir / "notes.txt"
    notes.write_text("some notes")
    fsm.add_file_to_global_helper(str(notes), description="test desc")

    # 3. get_file_description_from_helper → "test desc".
    assert fsm.get_file_description_from_helper(str(notes)) == "test desc"

    # 4. get_global_helper_as_context → non-empty, contains filename and description.
    ctx = fsm.get_global_helper_as_context()
    assert "notes.txt" in ctx
    assert "test desc" in ctx

    # 5. add_file_to_global_helper on non-existent path → FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        fsm.add_file_to_global_helper(str(memory_dir / "ghost.txt"), description="x")

    # 6. Delete file; cleanup_deleted_files() → returns 1; entry removed.
    notes.unlink()
    removed = fsm.cleanup_deleted_files()
    assert removed == 1
    assert fsm.get_file_description_from_helper(str(notes)) is None

    # 7. Reload FileSystemManager from same dirs → entry no longer present.
    fsm2 = FileSystemManager(memory_dir=str(memory_dir))
    assert "notes.txt" not in fsm2.get_global_helper_as_context()


# ---------------------------------------------------------------------------
# 8. test_filesystem_manager_predefined_file_info
# ---------------------------------------------------------------------------

def test_filesystem_manager_predefined_file_info(tmp_path, monkeypatch):
    import tools.file_manager as fm_module

    monkeypatch.setattr(fm_module, "DATA_FILES_DIR", str(tmp_path))

    from tools.file_manager import FileSystemManager

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    fsm = FileSystemManager(memory_dir=str(memory_dir))

    # 1. Non-existent file → FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        fsm.add_predefined_file_info("nonexistent.txt", "desc")

    # 2. Write file; call with relative filename → no exception; entry in _file_index.
    info = memory_dir / "info.txt"
    info.write_text("info content")
    fsm.add_predefined_file_info("info.txt", "first desc")
    assert "info.txt" in fsm._file_index

    # 3. Call again with update_existing=False → description unchanged.
    fsm.add_predefined_file_info("info.txt", "new desc", update_existing=False)
    assert fsm._file_index["info.txt"]["description"] == "first desc"

    # 4. Call with update_existing=True → description updated, last_updated refreshed.
    old_ts = fsm._file_index["info.txt"].get("last_updated")
    time.sleep(0.02)
    fsm.add_predefined_file_info("info.txt", "updated desc", update_existing=True)
    assert fsm._file_index["info.txt"]["description"] == "updated desc"
    assert fsm._file_index["info.txt"].get("last_updated") != old_ts

    # 5. Absolute path → normalized and stored correctly.
    fsm.add_predefined_file_info(str(info), "abs desc", update_existing=True)
    assert "info.txt" in fsm._file_index


# ---------------------------------------------------------------------------
# 13. test_tool_definitions_integrity
# ---------------------------------------------------------------------------

def test_tool_definitions_integrity():
    from tools.tool_register import (
        DEFINED_TOOLS,
        DEFINED_TOOLS_DICT,
        get_file_content,
        get_file_list,
    )

    # 1. Both non-empty; lengths equal.
    assert len(DEFINED_TOOLS) > 0
    assert len(DEFINED_TOOLS_DICT) == len(DEFINED_TOOLS)

    # 2. Each tool has .name and .description.
    for t in DEFINED_TOOLS:
        assert hasattr(t, "name")
        assert hasattr(t, "description")

    # 3. DEFINED_TOOLS_DICT keys match tool.name.
    for t in DEFINED_TOOLS:
        assert t.name in DEFINED_TOOLS_DICT
        assert DEFINED_TOOLS_DICT[t.name] is t

    # 4. get_file_list() → returns a list (may be empty).
    result = get_file_list.func()
    assert isinstance(result, list)

    # 5. get_file_content("nonexistent_file.txt") → error string, not exception.
    result = get_file_content.func("nonexistent_file.txt")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 14. test_datasheet_tools_pipeline [integration-lite]
# ---------------------------------------------------------------------------

@pytest.mark.integration_lite
def test_datasheet_tools_pipeline(tmp_path):
    from tools.tool_register import (
        calculate_datasheet_statistics,
        get_datasheet_chunk,
        get_full_dataframe_string_tool,
    )

    # 1. Real CSV with numeric column.
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,value\nalice,10\nbob,20\n")

    # 2. get_full_dataframe_string_tool → non-empty string.
    result = get_full_dataframe_string_tool.func({"file_path": str(csv_path)})
    assert isinstance(result, str)
    assert result

    # 3. get_datasheet_chunk with rows=[0] → single-row string.
    result = get_datasheet_chunk.func({"file_path": str(csv_path), "rows": [0]})
    assert isinstance(result, str)

    # 4. calculate_datasheet_statistics → dict with requested stat keys for column.
    result = calculate_datasheet_statistics.func(
        {"file_path": str(csv_path), "columns": "value", "stats": ["mean", "max"]}
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert "mean" in result["value"]
    assert "max" in result["value"]

    # 5. Without file_path (reuse cached df) → same result.
    result2 = calculate_datasheet_statistics.func(
        {"columns": "value", "stats": ["mean", "max"]}
    )
    assert result2 == result

    # 6. Unsupported extension → error string, not exception.
    exe_path = tmp_path / "data.exe"
    exe_path.write_text("fake")
    try:
        result3 = get_full_dataframe_string_tool.func({"file_path": str(exe_path)})
        assert isinstance(result3, str)
    except Exception as exc:
        pytest.fail(f"Expected error string, got exception: {exc}")
