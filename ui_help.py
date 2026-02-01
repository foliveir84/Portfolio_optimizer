import streamlit as st


def show_glossary():
    """Exibe um glossário explicativo dos termos técnicos e gráficos."""

    st.markdown("---")
    with st.expander("ℹ️ Glossário: Termos e Gráficos"):
        tab1, tab2 = st.tabs(
            ["📄 Termos da Tabela", "📊 Interpretação dos Gráficos"])

        with tab1:
            st.markdown("""
            **Colunas da Tabela:**
            * **Vol (Volume):** Unidades vendidas nos últimos 6 meses.
            * **Margem Real:** O lucro unitário atual (Stock existente).
            * **Margem Teórica:** O lucro unitário de reposição (Preço de compra hoje).
            * **Nova Margem:** O lucro unitário da sugestão (Novo produto).
            * **Ganho Est. (Ganho Estimado):** Quanto dinheiro a mais teria ganho se tivesse vendido a sugestão.
            * **Delta Preço:** Diferença para o utente (`+` paga mais, `-` poupa).
            """)

        with tab2:
            st.markdown("""
            **1. Matriz de Decisão Estratégica (4 Quadrantes)**
            Cruza o *Volume de Vendas* (Eixo X) com o *Ganho Unitário Adicional* (Eixo Y).
            
            * **💎 OURO (Canto Superior Direito):** Produtos com muita saída e grande aumento de margem. **Ação:** Troca obrigatória e imediata. Prioridade máxima da equipa.
            * **🐄 CASH COW (Canto Inferior Direito):** Produtos que vendem muito, mas o ganho extra por unidade é pequeno (ex: +0.05€). **Ação:** O lucro vem da quantidade. Trocar, mas sem urgência crítica.
            * **🎯 NICHO (Canto Superior Esquerdo):** Produtos que vendem pouco, mas cada troca dá um lucro enorme (ex: +5.00€). **Ação:** Garantir stock, pois cada venda conta muito.
            * **❓ INTERROGAÇÃO (Canto Inferior Esquerdo):** Pouco volume e pouco ganho. **Ação:** Baixa prioridade.
            
            ---
            
            **2. Ponte de Margem (Antes vs. Depois)**
            Mostra o impacto financeiro direto no Top 10 produtos.
            
            * **Barra Cinzenta:** O lucro total que teve com o produto atual.
            * **Barra Colorida:** O lucro total que *poderia ter tido* com a sugestão.
            * **Objetivo:** Visualizar o "salto" de rentabilidade. Se a barra colorida for o dobro da cinzenta, justifica qualquer esforço de mudança.
            """)
