import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import clean_model_output, parse_tool_call, run_turn
from toolkit import (
    TOOL_DEFINITIONS,
    auto_select_skills,
    direct_tool_call_for_request,
    narrow_tools_for_request,
    tool_calculator,
    tool_disk_usage,
    tool_list_notes,
    tool_percentage,
    tool_read_file,
    tool_recall,
    tool_remember,
    tool_search_files,
    tool_write_file,
    tools_for_skills,
)


class ToolkitTests(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(tool_calculator("0.12 * 4580"), "549.6")
        self.assertEqual(tool_percentage(12, 4580), "549.6")
        with self.assertRaises(ValueError):
            tool_calculator("__import__('os').system('id')")

    def test_skill_router(self):
        self.assertEqual(auto_select_skills("What is 4 * 9?"), ["core"])
        self.assertIn("web", auto_select_skills("Search the web for weather news"))
        self.assertIn("files", auto_select_skills("Find TODO in Python files"))
        self.assertIn("system", auto_select_skills("How much disk space is free?"))
        self.assertIn("memory", auto_select_skills("Remember this note"))
        self.assertIn(
            "web",
            auto_select_skills(
                "What USA state has the largest population as of 2026?"
            ),
        )
        self.assertNotIn("shell", auto_select_skills("How much disk space is free?"))

    def test_current_fact_gets_only_web_search(self):
        from toolkit import build_system_prompt

        tools = tools_for_skills(["core", "web"])
        narrowed = narrow_tools_for_request(
            "What USA state has the largest population as of 2026?", tools
        )
        self.assertIn("web_search", narrowed)
        self.assertNotIn("fetch_url", narrowed)
        self.assertNotIn("fetch_json", narrowed)
        self.assertNotIn("weather", narrowed)
        prompt = build_system_prompt(narrowed)
        self.assertIn('<function name="web_search">', prompt)
        self.assertIn("MUST call web_search", prompt)

    @patch("toolkit._http_get")
    def test_jina_bing_search_parser(self, mock_get):
        from toolkit import _normalize_search_query, _search_jina_bing

        mock_get.return_value = (
            "1.   ## [California population](https://example.com/california)\n\n"
            "California is the most populous U.S. state.\n\n"
            "2.   ## [State estimates](https://example.com/estimates)\n\n"
            "Annual population estimates.\n"
        )
        rows = _search_jina_bing("largest state population")
        self.assertEqual(len(rows), 2)
        self.assertIn("California", rows[0])
        self.assertIn("https://example.com/california", rows[0])
        self.assertEqual(
            _normalize_search_query(
                "What USA state has the largest population as of 2026?"
            ),
            "largest US state population 2026",
        )

    def test_disk_request_gets_deterministic_direct_tool(self):
        tools = tools_for_skills(["core", "system"])
        self.assertEqual(
            direct_tool_call_for_request("How much disk space is free?", tools),
            ("disk_usage", {"path": "/"}),
        )
        self.assertIn("free=", tool_disk_usage("/"))

    def test_memory_tools_are_narrowed_by_intent(self):
        memory_tools = tools_for_skills(["core", "memory"])
        remember = narrow_tools_for_request("Remember this note", memory_tools)
        recall = narrow_tools_for_request("Recall my API note", memory_tools)
        listing = narrow_tools_for_request("List my notes", memory_tools)
        self.assertIn("remember", remember)
        self.assertNotIn("recall", remember)
        self.assertIn("recall", recall)
        self.assertNotIn("remember", recall)
        self.assertIn("list_notes", listing)
        self.assertNotIn("remember", listing)

    def test_file_tools_and_workspace_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MINICPM_WORKSPACE": directory}, clear=False):
                message = tool_write_file("notes/demo.txt", "alpha\nTODO: beta")
                self.assertIn("Wrote", message)
                self.assertEqual(tool_read_file("notes/demo.txt"), "alpha\nTODO: beta")
                matches = tool_search_files("todo", pattern="*.txt")
                self.assertIn("notes/demo.txt:2", matches)
                with self.assertRaises(PermissionError):
                    tool_read_file("/etc/passwd")

    def test_persistent_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MINICPM_WORKSPACE": directory}, clear=False):
                tool_remember("server", "Use port 8088")
                self.assertIn("port 8088", tool_recall("server"))
                self.assertIn("server", tool_list_notes())
                tool_remember("Local Test API", "uses port 8088")
                self.assertIn("port 8088", tool_recall("local test API port"))

    def test_xml_and_json_tool_call_parsing(self):
        tools = tools_for_skills(["core"])
        xml = '<function name="calculator"><param name="expression">2 &lt; 3</param></function>'
        self.assertEqual(parse_tool_call(xml, tools), ("calculator", {"expression": "2 < 3"}))
        json_call = '<tool_call>{"name":"calculator","arguments":{"expression":"6*7"}}</tool_call>'
        self.assertEqual(
            parse_tool_call(json_call, tools),
            ("calculator", {"expression": "6*7"}),
        )
        self.assertIsNone(parse_tool_call('<function name="shell"></function>', tools))

    def test_tool_registry_functions_are_callable(self):
        self.assertTrue(TOOL_DEFINITIONS)
        self.assertTrue(all(callable(tool["fn"]) for tool in TOOL_DEFINITIONS.values()))

    @patch("agent.chat")
    def test_direct_tool_result_bypasses_second_model_call(self, mock_chat):
        mock_chat.return_value = (
            '<function name="percentage"><param name="percent">12</param>'
            '<param name="value">4580</param></function>'
        )
        answer, _ = run_turn(
            [], "What is 12% of 4580?", requested_skills=["core"], verbose=False
        )
        self.assertEqual(answer, "549.6")
        mock_chat.assert_called_once()

    @patch("agent.chat")
    def test_disk_fast_path_never_calls_model(self, mock_chat):
        answer, _ = run_turn(
            [], "How much disk space is free?", requested_skills=["core", "system"], verbose=False
        )
        self.assertIn("free=", answer)
        mock_chat.assert_not_called()

    @patch("agent.chat")
    def test_unfinished_thinking_is_retried_then_hidden(self, mock_chat):
        mock_chat.return_value = "<think>still reasoning without an end"
        answer, _ = run_turn([], "Explain something", requested_skills=["core"], verbose=False)
        self.assertNotIn("still reasoning", answer)
        self.assertNotIn("<think>", answer)
        self.assertEqual(mock_chat.call_count, 2)

    def test_clean_model_output_detects_unfinished_thinking(self):
        visible, unfinished = clean_model_output("<think>partial")
        self.assertEqual(visible, "")
        self.assertTrue(unfinished)


class NewSkillPackTests(unittest.TestCase):
    def test_unit_convert(self):
        from toolkit import tool_unit_convert

        self.assertIn("6.21371 mi", tool_unit_convert(10, "km", "miles"))
        self.assertIn("212 f", tool_unit_convert(100, "celsius", "fahrenheit"))
        self.assertIn("1000 mb", tool_unit_convert(1, "gigabyte", "megabytes"))
        with self.assertRaises(ValueError):
            tool_unit_convert(1, "kg", "miles")
        with self.assertRaises(ValueError):
            tool_unit_convert(1, "furlong", "m")

    def test_date_tools(self):
        import datetime as dt

        from toolkit import tool_date_add, tool_days_between

        expected = (dt.date(2026, 7, 10) - dt.date(2026, 1, 1)).days
        self.assertIn(str(expected), tool_days_between("2026-01-01", "2026-07-10"))
        self.assertIn("2026-01-31", tool_date_add("2026-01-01", 30))
        self.assertIn("2025-12-02", tool_date_add("2026-01-01", -30))

    def test_csv_tools(self):
        from toolkit import tool_csv_column_stats, tool_csv_inspect

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MINICPM_WORKSPACE": directory}, clear=False):
                Path(directory, "sales.csv").write_text(
                    "sku,qty,price\nA-1,2,$10.50\nB-2,1,$5.00\nA-1,3,$10.50\n"
                )
                inspected = tool_csv_inspect("sales.csv")
                self.assertIn("sku, qty, price", inspected)
                self.assertIn("Data rows: 3", inspected)
                stats = tool_csv_column_stats("sales.csv", "price")
                self.assertIn("sum 26", stats)
                with self.assertRaises(ValueError):
                    tool_csv_column_stats("sales.csv", "missing")

    def test_git_tools(self):
        import subprocess

        from toolkit import tool_git_log, tool_git_status

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MINICPM_WORKSPACE": directory}, clear=False):
                subprocess.run(
                    ["git", "init", "-q"], cwd=directory, check=True, timeout=15
                )
                self.assertIn("##", tool_git_status())
                self.assertTrue(tool_git_log(5))

    def test_router_reaches_new_packs(self):
        self.assertIn("convert", auto_select_skills("Convert 10 km to miles"))
        self.assertIn("convert", auto_select_skills("How many days until 2026-12-25?"))
        self.assertIn("finance", auto_select_skills("What is the AAPL stock price?"))
        self.assertIn("finance", auto_select_skills("How much is bitcoin right now?"))
        self.assertIn("git", auto_select_skills("Show me the git status"))
        self.assertIn("data", auto_select_skills("Summarize the price column in sales.csv"))
        self.assertIn("network", auto_select_skills("What is my public ip?"))
        self.assertIn("network", auto_select_skills("Is example.com up?"))
        self.assertNotIn("finance", auto_select_skills("What is 4 * 9?"))

    def test_new_tools_are_registered(self):
        packs = tools_for_skills(["convert", "finance", "git", "data", "network"])
        self.assertEqual(len(packs), 15)
        self.assertTrue(all(callable(tool["fn"]) for tool in packs.values()))


class GeneralKnowledgeTests(unittest.TestCase):
    def test_question_needs_tools(self):
        from toolkit import question_needs_tools

        self.assertFalse(
            question_needs_tools("What is the capital of California?", ["core"])
        )
        self.assertFalse(question_needs_tools("Who wrote Moby Dick?", ["core"]))
        self.assertTrue(question_needs_tools("What is 4 * 9?", ["core"]))
        self.assertTrue(question_needs_tools("What time is it?", ["core"]))
        self.assertTrue(
            question_needs_tools("What is the capital of California?", ["core", "web"])
        )

    @patch("agent.chat")
    def test_general_question_skips_tool_catalog(self, mock_chat):
        mock_chat.return_value = "The capital of California is Sacramento."
        answer, _ = run_turn(
            [], "What is the capital of California?", verbose=False
        )
        self.assertIn("Sacramento", answer)
        mock_chat.assert_called_once()
        system_prompt = mock_chat.call_args[0][0][0]["content"]
        self.assertNotIn("<function", system_prompt)
        self.assertNotIn("calculator", system_prompt)


class RepeatedCallTests(unittest.TestCase):
    @patch("toolkit.tool_web_search")
    @patch("agent.chat")
    def test_repeated_call_triggers_clean_synthesis(self, mock_chat, mock_search):
        mock_search.return_value = "1. US states by population\nCalifornia is #1"
        search_call = (
            '<function name="web_search">'
            '<param name="query">largest US state population</param></function>'
        )
        mock_chat.side_effect = [
            search_call,
            search_call,  # model repeats itself instead of answering
            "California has the largest population.",  # synthesis turn
        ]
        answer, _ = run_turn(
            [],
            "What USA state has the largest population?",
            requested_skills=["core", "web"],
            verbose=False,
        )
        self.assertIn("California", answer)
        self.assertNotIn("Do not call", answer)
        # synthesis turn must carry no tool catalog to parrot
        synthesis_system = mock_chat.call_args[0][0][0]["content"]
        self.assertNotIn("<function", synthesis_system)


if __name__ == "__main__":
    unittest.main()
