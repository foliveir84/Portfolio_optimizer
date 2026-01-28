# 💊 Otimizador de Margem Farmacêutica (CNPME) - Documentação do Projeto

**Estado Atual:** 🟢 Funcional / Estável / Interface "PharmaTouch"
**Data da Última Revisão:** 28/01/2026

---

## 1. Visão Geral e Objetivo
Sistema de Business Intelligence para farmácias comunitárias focado na **maximização da margem de lucro** através da substituição inteligente de genéricos. O sistema cruza dados reais de vendas com o portefólio nacional (Infarmed) e acordos comerciais (Descontos), sugerindo trocas que respeitam o grupo terapêutico (CNPEM) e a sensibilidade ao preço do utente.

## 2. Arquitetura Técnica

### Estrutura de Ficheiros
*   **`app.py`:** "Cérebro" da aplicação. Contém toda a lógica de ingestão, processamento, cálculo e interface. Substituiu os antigos scripts auxiliares (`process_*.py`) para permitir processamento dinâmico "on-the-fly".
*   **`ui_style.py`:** Módulo de design. Define o tema "PharmaTouch Glass" (CSS, cores, métricas).
*   **`requirements.txt`:** Dependências estritas (`streamlit`, `pandas`, `openpyxl`, `xlrd`).
*   **`.gitignore`:** Segurança. Bloqueia o upload de dados sensíveis (`*.csv`, `*.xls`, `*.txt`).

### Input de Dados
O sistema ingere 3 ficheiros críticos na barra lateral:
1.  **Vendas (Infoprex):** `.txt` ou `.csv`. Fonte da verdade sobre o que se vende e o stock atual.
2.  **Mestre (Infarmed):** `.xls`. Fonte oficial de PVPs, comparticipações e grupos homogéneos.
3.  **Descontos:** `.xlsx` (Colunas: `CNP`, `DESC`). Fonte das condições comerciais negociadas.

---

## 3. Lógica de Negócio (O "Coração" do Algoritmo)

### A. O Dilema do "Stock de Campanha"
Um dos maiores desafios resolvidos foi evitar sugestões erradas quando a farmácia tem stock comprado em campanha (custo muito baixo) vs. uma nova opção de mercado.

**Solução Implementada (Dual Check):**
O sistema calcula dois tipos de margem para cada produto:
1.  **Margem Real (Stock):** `(PVP/1.06) - PCU_Médio_Atual`. Reflete o lucro de vender o que está na prateleira.
2.  **Margem Teórica (Reposição):** `(PVP/1.06) - (PVA * (1 - Desconto_Negociado))`. Reflete o lucro de comprar este produto *hoje*.

**Matriz de Decisão:**
*   **"Trocar Já 🔄":** A Nova Opção tem uma margem *superior* à Margem Real do Stock atual. (Ganho imediato absoluto).
*   **"Esgotar 📉":** A Nova Opção é pior que o Stock atual (porque foi comprado barato), mas é *melhor* que a Margem Teórica do produto atual. (Ação: Vender o stock todo e, na próxima encomenda, mudar de marca).

### B. Gestão de Preços (Regimes)
*   **Switch Utente/Pensionista:** O sistema ajusta dinamicamente a coluna de preço (`Preço Utente` vs `Preço Pensionistas`) e o título das colunas, essencial para produtos onde o PVP5 altera drasticamente o valor a pagar.
*   **Tolerância:** Input livre para definir quanto a mais o utente aceita pagar (ex: +0.50€).

---

## 4. Correções Críticas e Estabilização

Durante o desenvolvimento, foram resolvidos vários pontos de falha:

1.  **Duplicação de Dados (Infoprex):**
    *   *Problema:* O ficheiro de vendas continha múltiplas linhas para o mesmo produto devido a diferentes "Localizações" ou datas.
    *   *Solução:* Implementado filtro automático que deteta a `DUV` (Data Última Venda) mais recente e filtra apenas a `LOCALIZACAO` correspondente.

2.  **Robustez na Leitura de Ficheiros:**
    *   *Problema:* Erros de `UnicodeDecodeError` e `ParserError` devido a formatos antigos (`latin-1`, separadores `	` vs `;`).
    *   *Solução:* Loop de tentativa/erro inteligente que testa combinações de *Encodings* e *Separadores* até encontrar uma estrutura tabular válida (>1 coluna).

3.  **Desaparecimento de Dados (KeyErrors):**
    *   *Problema:* O `merge` do Pandas criava sufixos (`_x`, `_y`) destruindo a coluna chave `CNPEM`.
    *   *Solução:* Seleção explícita de colunas (`sales_cols_keep`) antes do merge e normalização forçada dos nomes (`Substância Ativa` -> `GRUPOHOMOGENEO`, `Nº registo` -> `N_REGISTO`) logo na ingestão.

4.  **Inconsistência de Tipos (IDs):**
    *   *Solução:* Limpeza agressiva de IDs (`.astype(str).str.split('.').str[0]`) para garantir que `12345` (Excel) casa com `12345.0` (CSV).

---

## 5. Pipeline de Dados (Fluxo)

1.  **Ingestão:** Upload dos 3 ficheiros.
2.  **Normalização:** Renomeação de colunas para padrão interno.
3.  **Enriquecimento Mestre:** Cálculo de `PVA` (fórmula oficial) e `Margem Teórica` (com ficheiro Descontos).
4.  **Limpeza Vendas:** Filtro de Localização + Cálculo `Margem Real`.
5.  **Merge:** `Inner Join` entre Vendas e Mestre (apenas produtos vendidos são analisados).
6.  **Análise ABC:** Agrupamento por `CNPEM`, ordenação por Margem Total e classificação Pareto.
7.  **Simulação:** Iteração linha a linha com lógica comparativa e renderização visual.

---

## 6. Notas Finais
Este projeto está pronto para produção local. A separação entre Lógica (`app.py`) e Estilo (`ui_style.py`) facilita a manutenção futura. O sistema de leitura é agnóstico a pequenas variações de formato do Infoprex, garantindo longevidade.
