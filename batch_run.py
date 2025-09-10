import argparse
import os
from types import SimpleNamespace

from interface import run_quiz_ui


def main():
    parser = argparse.ArgumentParser(description="Run quiz analysis headlessly (no Gradio)")
    parser.add_argument("input_json", help="Path to quiz JSON file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and write logs to outputs/")
    parser.add_argument("--timeout", type=float, default=None, help="Per-question generation timeout in seconds")
    parser.add_argument("--skip-charts", action="store_true", help="Skip chart generation to avoid backend issues")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_json)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Quiz file not found: {input_path}")

    # Mimic gr.File object minimal interface: .name
    file_like = SimpleNamespace(name=input_path)

    (
        report,
        main_report_filename,
        mcq_file,
        writing_file,
        chart_file,
        summary_text,
        debug_logs,
    ) = run_quiz_ui(file_like, debug=args.debug, timeout_s=args.timeout, skip_charts=args.skip_charts)

    print("Analysis completed. Outputs:")
    print(f"- Complete report: {main_report_filename}")
    print(f"- MCQ detailed analysis: {mcq_file}")
    print(f"- Writing detailed analysis: {writing_file}")
    print(f"- Comprehensive chart: {chart_file}")
    if args.debug and debug_logs:
        print("\n--- Debug logs (tail) ---")
        print("\n".join(debug_logs.splitlines()[-50:]))


if __name__ == "__main__":
    main()


