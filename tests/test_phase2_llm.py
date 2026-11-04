from src.phase2.llm.interpreter import SopDiagnosisInterpreter


def test_llm_fallback(monkeypatch):
    interpreter = SopDiagnosisInterpreter()
    monkeypatch.setattr(interpreter, "_ollama_available", lambda: False)
    resp = interpreter.generate("teste")
    assert resp.provider == "fallback"
