import zipfile
from pathlib import Path

from rag import cli


def test_deployment_archive_excludes_secrets_and_research(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / ".streamlit").mkdir()
    (tmp_path / ".streamlit" / "secrets.toml").write_text("API_KEY=secret")
    (tmp_path / ".streamlit" / "config.toml").write_text("[server]")
    (tmp_path / "app.py").write_text("print('app')")
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "private.json").write_text("private")
    cli.package()
    with zipfile.ZipFile(tmp_path / "dist" / "evidence-lab-deploy.zip") as archive:
        names = archive.namelist()
        assert "app.py" in names
        assert all("secret" not in name and "research" not in name for name in names)


def test_no_local_model_runtime_dependencies():
    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text().lower()
    for forbidden in ["torch", "sentence-transformers", "ollama", "transformers"]:
        assert forbidden not in requirements
