import streamlit as st

def show_glossary():
    """Exibe um glossário explicativo dos termos técnicos da tabela de sugestões."""
    
    st.markdown("---")
    with st.expander("ℹ️ Glossário: Como interpretar os resultados"):
        st.markdown("""
        Esta tabela ajuda a decidir quando substituir um medicamento por outro mais rentável. Aqui está o significado de cada coluna:

        *   **Vol (Volume):** Quantidade total de unidades vendidas nos últimos 6 meses (baseado no ficheiro Infoprex).
        *   **Pr. Ut. At. / Pr. Pen. At.:** Preço que o utente (ou pensionista) paga atualmente pelo medicamento que tem em stock.
        *   **Pr. Ut. Novo / Pr. Pen. Novo:** Preço que o utente passaria a pagar se mudasse para o produto sugerido.
        *   **Margem Real:** O lucro unitário que está a ter **neste momento** com o stock que tem na prateleira (PVP s/ IVA - Preço de Custo Real).
        *   **Margem Teórica:** O lucro unitário que teria se comprasse o produto atual **hoje** (PVP s/ IVA - PVA c/ Desconto Comercial). Serve para comparar se o seu produto atual ainda é competitivo.
        *   **Nova Margem:** O lucro unitário que terá com o produto sugerido (PVP s/ IVA - PVA c/ Desconto Comercial).
        *   **Delta Preço:** A diferença de preço para o utente. 
            *   *Exemplo:* `+0.20€` significa que o utente paga mais 20 cêntimos; `-0.10€` significa que poupa 10 cêntimos.
        *   **Ganho Est. (Ganho Estimado):** O lucro extra total que a farmácia teria ganho nos últimos 6 meses se tivesse vendido a sugestão em vez do produto atual.
        
        ---
        **Dica de Ação:**
        *   **Trocar Já 🔄:** A nova opção é tão boa que ganha mais dinheiro do que vendendo o stock que já pagou.
        *   **Esgotar 📉:** O stock atual foi comprado em condições muito boas (ex: campanha). Venda tudo o que tem e só mude de marca na próxima encomenda.
        """)
