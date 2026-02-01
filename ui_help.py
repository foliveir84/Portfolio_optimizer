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
            **1. Matriz de Saúde do Portfólio (Interativa)**
            Esta ferramenta visualiza a *saúde atual* do seu stock.
            
            * **Eixo Vertical (Altura):** Representa o **Lucro Real Total** que o produto gera hoje. Quanto mais alto, mais importante esse produto é para a farmácia.
            * **Tamanho da Bola:** O potencial de ganho extra. Bolas grandes significam grandes oportunidades.
            * **Código de Cores (O Semáforo):**
                * 🔴 **VERMELHO (Crítico):** O produto vende bem, mas a sua margem atual é muito inferior à alternativa. **Ação:** Trocar imediatamente.
                * 🟡 **AMARELO (Atenção):** Existe margem para melhorar, mas não é urgente.
                * 🟢 **VERDE (Otimizado):** O produto já é a melhor opção do mercado. **Ação:** Garantir stock e evitar ruturas.
            
            **Objetivo do Gestor:** "Arrefecer" o gráfico. Transformar as bolas vermelhas do topo em bolas verdes, mantendo-as na parte superior (alto lucro).

            ---
            
            **2. Ponte de Margem (Top 10)**
            Mostra o "salto" financeiro possível nos 10 produtos mais críticos.
            
            * **Barra Cinzenta:** O que já ganha hoje.
            * **Barra Azul:** O dinheiro "extra" que ficaria na farmácia se trocasse a marca.
            """)
