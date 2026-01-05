# Ciclo de Estudos — Web App (3 Tabelas / Abas Principais)

Web app em Streamlit com:

1. **Tabela 1 — Painel Geral de Performance e Execução**
2. **Tabela 2 — Revisões e Desempenho** (botões Google Calendar + lote)
3. **Tabela 3 — Acompanhamento de Evolução e Métricas**

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dados
SQLite em `data/app.db`.

## Como usar
Vá em **Configuração → Cadastros** e adicione:
- Matérias (Peso, Questões, Dificuldade)
- Assuntos por matéria
Depois registre estudos (isso gera revisões automáticas).
