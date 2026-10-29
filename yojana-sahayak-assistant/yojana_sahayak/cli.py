"""
Command-line interface for Yojana Sahayak.

Usage:
    python -m yojana_sahayak.cli --text "PM Kisan ke liye kaun eligible hai?"
    python -m yojana_sahayak.cli --voice
    python -m yojana_sahayak.cli --gradio
    python -m yojana_sahayak.cli --mcp
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Yojana Sahayak — Sovereign Voice Agent for Indian Government Schemes"
    )
    parser.add_argument("--text", type=str, help="Text query (skips ASR)")
    parser.add_argument("--voice", action="store_true", help="Interactive voice loop")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio web demo")
    parser.add_argument("--mcp", action="store_true", help="Run MCP server (stdio)")
    args = parser.parse_args()

    if args.mcp:
        from yojana_sahayak.mcp.server import main as mcp_main
        mcp_main()

    elif args.gradio:
        from yojana_sahayak.demo import launch_gradio
        launch_gradio()

    elif args.text:
        from yojana_sahayak.agent.pipeline import YojanaPipeline
        pipeline = YojanaPipeline()
        result = pipeline.run(text_input=args.text, speak=False)
        print(f"\nAnswer: {result.answer}")

    elif args.voice:
        from yojana_sahayak.agent.pipeline import YojanaPipeline
        from yojana_sahayak.asr.whisper import record_mic
        pipeline = YojanaPipeline()
        print("\n" + "=" * 55)
        print("  YOJANA SAHAYAK — Voice Assistant")
        print("  Ask about any Indian government scheme")
        print("  Press Ctrl+C to exit")
        print("=" * 55)

        while True:
            input("\n  Press ENTER to start speaking...")
            audio_path = record_mic()
            pipeline.run(audio_path=audio_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
