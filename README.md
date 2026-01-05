# Ciclo de Estudos — Multi-Concurso (Fix 2)

**Correção do erro no Streamlit Cloud:**
Se você já tinha um banco antigo, a tabela `subjects` pode não ter a coluna `exam_id`.
Este app faz **migração automática** (adiciona `exam_id` e cria um concurso padrão).

Deploy: substitua o `app.py` e `requirements.txt` no seu repositório do GitHub.
