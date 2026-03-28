"""启动入口"""

from src.orchestrator import build_controller
from src.tui.cmd_processor import execute_shell_stream
from src.tui.application import AgentCLI

def main():
    controller = build_controller(shell_executor=execute_shell_stream)
    app = AgentCLI(input_handler=controller)
    app.run()


if __name__ == "__main__":
    main()
