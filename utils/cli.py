import typer
from .run import run_task

app = typer.Typer(help="PhenoAssistant CLI")

@app.command("run")
def run(task: str, env: str = ".env.yaml"):
    """Run the manager on a free-form task string."""
    run_task(task=task, env_path=env)

if __name__ == "__main__":
    app()
