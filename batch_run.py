import argparse
import os
from types import SimpleNamespace

from interface import run_quiz_ui


def main():
    parser = argparse.ArgumentParser(description="Run quiz analysis headlessly (no Gradio)")
    parser.add_argument("input_json", help="Path to quiz JSON file")
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
    ) = run_quiz_ui(file_like)

    print("Analysis completed. Outputs:")
    print(f"- Complete report: {main_report_filename}")
    print(f"- MCQ detailed analysis: {mcq_file}")
    print(f"- Writing detailed analysis: {writing_file}")
    print(f"- Comprehensive chart: {chart_file}")


if __name__ == "__main__":
    main()


