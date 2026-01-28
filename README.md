# 💊 Otimizador de Margem Farmacêutica (CNPME)

Ferramenta de Business Intelligence para farmácias, desenvolvida em Python/Streamlit. Analisa vendas, cruza com o portefólio nacional (Infarmed) e sugere trocas de produtos para maximizar a margem, respeitando o agrupamento CNPEM e a tolerância de preço do utente.

## 🚀 Funcionalidades

- **Análise ABC/XYZ:** Identifica os grupos terapêuticos (CNPEM) mais valiosos.
- **Simulador de Troca:**
  - Compara a **Margem Real** (Stock atual) vs **Margem Teórica** (Reposição de mercado).
  - Sugere "Trocar Já" ou "Esgotar Stock" baseado na rentabilidade.
  - Filtra por **Regime** (Utente vs Pensionista).
- **Interface Moderna:** Design "PharmaTouch Glass" com modo escuro.

## 📦 Instalação

1. **Clonar o repositório:**
   ```bash
   git clone <teu-repo-url>
   cd Analise_Genericos
   ```

2. **Criar ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Como Usar

1. Executar a aplicação:
   ```bash
   streamlit run app.py
   ```

2. Na barra lateral, fazer upload dos 3 ficheiros obrigatórios:
   - **Vendas:** Ficheiro `.txt` ou `.csv` (Exportação Infoprex).
   - **Mestre:** Ficheiro `.xls` oficial do Infarmed.
   - **Descontos:** Ficheiro `.xlsx` com colunas `CNP` e `DESC`.

## 🛡️ Estrutura de Ficheiros

- `app.py`: Lógica principal da aplicação.
- `ui_style.py`: Definições de design e CSS.
- `.gitignore`: Garante que dados sensíveis não são enviados para o Git.
