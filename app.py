import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import download_infarmed_data
import ui_help
import ui_style

# --- Configuração Inicial ---
st.set_page_config(page_title="Análise Genéricos ABC/XYZ", layout="wide")

# ==============================================================================
# 0. SISTEMA DE LOGIN (Versão Simples via Secrets)
# ==============================================================================


def check_login():
    """Gere a autenticação usando o ficheiro .streamlit/secrets.toml."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("### 🔐 Acesso Restrito")

    # Tenta carregar utilizadores
    try:
        USERS = st.secrets["users"]
    except Exception:
        st.error("Erro: Configure o ficheiro .streamlit/secrets.toml")
        st.stop()

    c1, c2 = st.columns([1, 2])
    with c1:
        username = st.text_input("Utilizador")
        password = st.text_input("Password", type="password")

        if st.button("Entrar"):
            if username in USERS and USERS[username] == password:
                st.session_state.authenticated = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("Credenciais inválidas")
    return False


if not check_login():
    st.stop()

# --- APP PRINCIPAL ---

master_path = "allPackages_python.xls"
if 'startup_check_done' not in st.session_state:
    if not os.path.exists(master_path):
        download_infarmed_data.download_infarmed_xls()
    st.session_state['startup_check_done'] = True

ui_style.init_session_state()
ui_style.apply_custom_style()

# ==============================================================================
# 1. LÓGICA DE DADOS
# ==============================================================================


def calcular_pva(pvp):
    """Fórmula oficial do PVA (Preço Venda Armazenista)."""
    if pd.isna(pvp):
        return 0.0
    if pvp <= 6.68:
        return ((pvp - 0.94) / 1.1475) + (0.004 * (pvp / 1.06))
    elif pvp <= 9.97:
        return ((pvp - 1.95) / 1.1460) + (0.004 * (pvp / 1.06))
    elif pvp <= 14.10:
        return ((pvp - 2.66) / 1.1436) + (0.004 * (pvp / 1.06))
    elif pvp <= 26.96:
        return ((pvp - 4.17) / 1.1393) + (0.004 * (pvp / 1.06))
    elif pvp <= 64.68:
        return ((pvp - 8) / 1.1316) + (0.004 * (pvp / 1.06))
    else:
        return ((pvp - 12.73) / 1.1051) + (0.004 * (pvp / 1.06))


@st.cache_data
def process_data(file_sales, file_master, file_discounts, master_mtime=None):
    try:
        # --- A. VENDAS (INFOPREX) ---
        file_sales.seek(0)
        try:
            df_sales = pd.read_csv(
                file_sales, sep='\t', encoding='latin-1', on_bad_lines='skip', dtype=str)
        except:
            df_sales = pd.read_csv(
                file_sales, sep=';', encoding='latin-1', on_bad_lines='skip', dtype=str)

        # Filtro Localização (Data mais recente)
        if 'DUV' in df_sales.columns and 'LOCALIZACAO' in df_sales.columns:
            df_sales['DUV_dt'] = pd.to_datetime(
                df_sales['DUV'], dayfirst=True, format='mixed', errors='coerce')
            valid = df_sales.dropna(subset=['DUV_dt'])
            if not valid.empty:
                max_loc = df_sales.loc[valid['DUV_dt'].idxmax(), 'LOCALIZACAO']
                df_sales = df_sales[df_sales['LOCALIZACAO'] == max_loc].copy()

        # Filtro Grupo Homogéneo
        col_grupo = next(
            (c for c in ['GRUPOHOMOGENEO', 'CODIGOHOMOGENEO'] if c in df_sales.columns), None)
        if col_grupo:
            df_sales = df_sales[df_sales[col_grupo].notna()]

        # Limpeza Numérica
        for c in ['PVP', 'PCU']:
            if c in df_sales.columns:
                df_sales[c] = pd.to_numeric(df_sales[c].str.replace(
                    ',', '.', regex=False), errors='coerce').fillna(0.0)

        # Vendas Totais
        v_cols = [c for c in [f'V({i})' for i in range(
            1, 7)] if c in df_sales.columns]
        for c in v_cols:
            df_sales[c] = pd.to_numeric(df_sales[c], errors='coerce').fillna(0)
        df_sales['VENDAS_6M'] = df_sales[v_cols].sum(axis=1)

        # Margens
        df_sales['MARGEM_REAL_UNIT'] = (
            (df_sales['PVP']/1.06) - df_sales['PCU']).round(2) if 'PVP' in df_sales.columns else 0.0
        df_sales['MARGEM_REAL_TOTAL'] = df_sales['MARGEM_REAL_UNIT'] * \
            df_sales['VENDAS_6M']
        df_sales['CPR'] = df_sales['CPR'].astype(
            str).str.split('.').str[0].str.strip()
        if col_grupo:
            df_sales['NOME_GRUPO_SALES'] = df_sales[col_grupo]

        # --- B. MESTRE (INFARMED) ---
        df_master = pd.read_excel(file_master)
        df_master.rename(columns={'Nº registo': 'N_REGISTO', 'Preço (PVP)': 'PVP', 'Preço Utente': 'P_UTENTE',
                         'Preço Pensionistas': 'P_PENSIONISTA', 'Nome do medicamento': 'Nome'}, inplace=True)
        df_master['N_REGISTO'] = pd.to_numeric(
            df_master['N_REGISTO'], errors='coerce').fillna(0).astype(int).astype(str)
        if 'CNPEM' in df_master.columns:
            df_master = df_master.dropna(subset=['CNPEM'])

        for c in ['PVP', 'P_UTENTE', 'P_PENSIONISTA']:
            if c in df_master.columns:
                if df_master[c].dtype == object:
                    df_master[c] = df_master[c].astype(
                        str).str.replace(',', '.', regex=False)
                df_master[c] = pd.to_numeric(
                    df_master[c], errors='coerce').fillna(0.0)
        df_master['PVA'] = df_master['PVP'].apply(calcular_pva)

        # --- C. DESCONTOS & LABORATORIOS ---
        df_desc = pd.read_excel(file_discounts)
        df_desc.columns = [c.upper().strip() for c in df_desc.columns]

        # Detectar colunas dinamicamente
        lab_col = next(
            (c for c in df_desc.columns if 'LAB' in c), 'LAB_DEFAULT')
        if lab_col == 'LAB_DEFAULT':
            df_desc['LAB_DEFAULT'] = 'Geral'
        cnp_col = next((c for c in df_desc.columns if 'CNP' in c), 'CNP')
        desc_col = next((c for c in df_desc.columns if 'DESC' in c), 'DESC')

        df_desc[cnp_col] = pd.to_numeric(
            df_desc[cnp_col], errors='coerce').fillna(0).astype(int).astype(str)
        df_desc[desc_col] = df_desc[desc_col].apply(lambda x: float(str(x).replace(
            ',', '.'))/100 if float(str(x).replace(',', '.')) > 1 else float(str(x).replace(',', '.')))

        # Merge
        df_master = pd.merge(df_master, df_desc[[
                             cnp_col, desc_col, lab_col]], left_on='N_REGISTO', right_on=cnp_col, how='left')
        df_master['DESC_COMERCIAL'] = df_master[desc_col].fillna(0.0)
        df_master['LABORATORIO_DESC'] = df_master[lab_col].fillna(
            'Outros/Sem Acordo')

        # Calcular Margem Teórica (Estrutural)
        df_master['MARGEM_TEORICA'] = (
            df_master['PVP']/1.06) - (df_master['PVA'] * (1 - df_master['DESC_COMERCIAL']))

        # --- D. MERGE FINAL ---
        cols_s = ['CPR', 'NOM', 'VENDAS_6M',
                  'MARGEM_REAL_UNIT', 'MARGEM_REAL_TOTAL']
        if 'NOME_GRUPO_SALES' in df_sales.columns:
            cols_s.append('NOME_GRUPO_SALES')
        df_final = pd.merge(
            df_sales[cols_s], df_master, left_on='CPR', right_on='N_REGISTO', how='inner')

        return df_final, df_master
    except Exception as e:
        return None, str(e)


def run_abc_analysis(df):
    if df.empty:
        return pd.DataFrame()
    g_col = 'NOME_GRUPO_SALES' if 'NOME_GRUPO_SALES' in df.columns else 'CNPEM'
    grouped = df.groupby('CNPEM').agg({'MARGEM_REAL_TOTAL': 'sum', 'VENDAS_6M': 'sum',
                                       'CPR': 'count', g_col: 'first'}).rename(columns={g_col: 'NOME_GRUPO'})
    grouped = grouped.sort_values('MARGEM_REAL_TOTAL', ascending=False)
    grouped['CLASSE_ABC'] = pd.cut(grouped['MARGEM_REAL_TOTAL'].cumsum(
    ) / grouped['MARGEM_REAL_TOTAL'].sum(), bins=[0, 0.8, 0.95, 1.0], labels=['A', 'B', 'C'])
    return grouped

# ==============================================================================
# 2. INTERFACE GRÁFICA
# ==============================================================================


# Barra Lateral
st.sidebar.markdown(
    f"👤 *Logado como: {st.session_state.get('username', 'Utilizador')}*")
if st.sidebar.button("Terminar Sessão"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.header("📁 Carregar Dados")
up_sales = st.sidebar.file_uploader("Vendas (Infoprex)", type=['txt', 'csv'])
st.sidebar.caption("Base de Dados Infarmed: " +
                   ("✅" if os.path.exists(master_path) else "⚠️"))

if st.sidebar.button("🔄 Atualizar BD"):
    download_infarmed_data.download_infarmed_xls()
    st.rerun()

up_desc = st.sidebar.file_uploader("Descontos (.xlsx)", type=['xlsx'])

# Processamento
if up_sales and os.path.exists(master_path) and up_desc:
    df, df_master = process_data(
        up_sales, master_path, up_desc, os.path.getmtime(master_path))

    if df is not None and not isinstance(df, str):
        # --- FILTRO DE LABORATÓRIOS ---
        all_labs = sorted(
            [str(x) for x in df_master['LABORATORIO_DESC'].unique() if pd.notna(x)])

        st.sidebar.divider()
        st.sidebar.markdown("### 🧪 Filtros")

        # Inicialização do estado do filtro
        if 'lab_filter_state' not in st.session_state:
            st.session_state['lab_filter_state'] = all_labs

        # Botões de controlo
        c1, c2 = st.sidebar.columns(2)
        if c1.button("✅ Todos", use_container_width=True):
            st.session_state['lab_filter_state'] = all_labs
        if c2.button("❌ Limpar", use_container_width=True):
            st.session_state['lab_filter_state'] = []

        # Widget Multiselect
        sel_labs = st.sidebar.multiselect(
            "Laboratórios:", all_labs, key='lab_filter_state')

        # Análise ABC
        abc = run_abc_analysis(df)
        class_a = abc[abc['CLASSE_ABC'] == 'A'].reset_index()
        class_a['LABEL'] = "#" + (class_a.index + 1).astype(str) + \
            " - " + class_a['NOME_GRUPO'].astype(str)

        st.subheader("🎯 Grupos Prioritários (Classe A)")
        if not class_a.empty:
            sel_label = st.selectbox("Selecione Grupo:", class_a['LABEL'])
            sel_cnpem = class_a[class_a['LABEL']
                                == sel_label]['CNPEM'].values[0]

            # Parametros Simulação
            st.divider()
            regime = st.sidebar.radio("Regime:", ["Geral", "R Especial"])
            p_col = 'P_PENSIONISTA' if regime == "R Especial" else 'P_UTENTE'
            lbl_p = "Pr. Pen." if regime == "R Especial" else "Pr. Ut."
            tol = st.sidebar.number_input("Tolerância (€)", 0.0, 5.0, 0.5)

            # Dados Filtrados
            products = df[df['CNPEM'] == sel_cnpem].copy(
            ).sort_values('VENDAS_6M', ascending=False)

            # Aplicar Filtro de Laboratórios ao Mercado (Candidatos)
            market = df_master[
                (df_master['CNPEM'] == sel_cnpem) &
                (df_master['LABORATORIO_DESC'].isin(sel_labs))
            ].copy()

            results = []
            for _, row in products.iterrows():
                if row['VENDAS_6M'] <= 0:
                    continue

                # Encontrar melhor candidato
                cands = market[
                    (market['MARGEM_TEORICA'] > row['MARGEM_TEORICA']) &
                    (market[p_col] <= (row[p_col] + tol)) &
                    (market['N_REGISTO'] != row['CPR'])
                ]

                if not cands.empty:
                    best = cands.sort_values(
                        'MARGEM_TEORICA', ascending=False).iloc[0]
                    gain_unit = best['MARGEM_TEORICA'] - \
                        row['MARGEM_REAL_UNIT']
                    gain_total = gain_unit * row['VENDAS_6M']

                    if gain_unit > 0:
                        action = "Trocar Já 🔄"
                    else:
                        action = "Esgotar 📉"

                    results.append({
                        'Produto': row['NOM'],
                        'Vol': row['VENDAS_6M'],
                        # Comparação de Margens
                        'Margem Real': row['MARGEM_REAL_UNIT'],
                        'Mg. Teórica At.': row['MARGEM_TEORICA'],
                        'Nova Margem': best['MARGEM_TEORICA'],
                        # Métricas
                        'Ganho Unit Delta': best['MARGEM_TEORICA'] - row['MARGEM_REAL_UNIT'],
                        'Ganho Est.': gain_total if gain_unit > 0 else 0,
                        'Ação': action,
                        'Pr. Atual': row[p_col],
                        'Pr. Novo': best[p_col],
                        'Delta Preço': best[p_col] - row[p_col],
                        'Sugestão': best['Nome'],
                        # Campos Ocultos para Gráficos
                        'Margem Total Atual': row['MARGEM_REAL_TOTAL'],
                        'Margem Total Nova': best['MARGEM_TEORICA'] * row['VENDAS_6M']
                    })

            if results:
                rdf = pd.DataFrame(results).sort_values(
                    'Ganho Est.', ascending=False)

                # Métricas de Topo
                c1, c2, c3 = st.columns(3)
                c1.metric("Potencial Ganho",
                          f"€ {rdf['Ganho Est.'].sum():,.2f}")
                c2.metric("Produtos Analisados", len(rdf))
                c3.metric("Sugestões Válidas", len(
                    rdf[rdf['Ação'] == 'Trocar Já 🔄']))

                # --- GRÁFICOS (BUSINESS INTELLIGENCE) ---
                st.markdown("### 📊 Análise Visual")
                chart_type = st.radio("Tipo de Gráfico:", [
                                      "Matriz Estratégica (4 Quadrantes)", "Ponte de Margem (Antes vs Depois)"], horizontal=True)

                if chart_type == "Matriz Estratégica (4 Quadrantes)":
                    fig, ax = plt.subplots(figsize=(10, 5))
                    # Tema Escuro/Neon
                    fig.patch.set_facecolor('#0e0b16')
                    ax.set_facecolor('#0e0b16')

                    x = rdf['Vol']
                    y = rdf['Ganho Unit Delta']
                    x_mid = x.mean() if not x.empty else 0
                    y_mid = y.mean() if not y.empty else 0

                    # Linhas Médias (Cruz)
                    ax.axvline(x_mid, color='#d900ff',
                               linestyle='--', alpha=0.3)
                    ax.axhline(y_mid, color='#d900ff',
                               linestyle='--', alpha=0.3)

                    # Quadrantes
                    ax.text(x.max(), y.max(), '💎 OURO\n(Alta Venda/Alto Ganho)',
                            color='#00e5ff', ha='right', va='top', fontsize=9, fontweight='bold')
                    ax.text(x.max(), y.min(), '🐄 CASH COW\n(Alta Venda/Baixo Ganho)',
                            color='white', ha='right', va='bottom', fontsize=8)
                    ax.text(x.min(), y.max(), '🎯 NICHO\n(Baixa Venda/Alto Ganho)',
                            color='white', ha='left', va='top', fontsize=8)

                    # Scatter
                    sizes = (rdf['Ganho Est.'] /
                             (rdf['Ganho Est.'].max() + 1) * 500) + 50
                    ax.scatter(x, y, s=sizes, c='#00e5ff',
                               alpha=0.7, edgecolors='white')

                    # --- NOVO: ETIQUETAS NOS TOP PRODUTOS ---
                    # Vamos etiquetar apenas os 7 melhores para não sujar o gráfico
                    top_items = rdf.head(7)

                    for idx, row in top_items.iterrows():
                        # Lógica simples para afastar o texto da bola
                        ax.annotate(
                            # Corta nomes muito longos
                            row['Produto'][:45] + '...',
                            (row['Vol'], row['Ganho Unit Delta']),
                            xytext=(5, 5), textcoords='offset points',
                            color='white', fontsize=5, alpha=0.9
                        )
                    # ----------------------------------------

                    ax.set_xlabel('Volume (Unidades)', color='white')
                    ax.set_ylabel('Ganho Unitário Extra (€)', color='white')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.spines['bottom'].set_visible(True)
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_visible(True)
                    ax.spines['left'].set_color('white')
                    ax.grid(color='gray', linestyle=':', alpha=0.3)
                    st.pyplot(fig)

                else:  # Ponte de Margem
                    top_n = rdf.head(10).sort_values(
                        'Ganho Est.', ascending=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0e0b16')
                    ax.set_facecolor('#0e0b16')

                    y_pos = np.arange(len(top_n))
                    height = 0.35

                    # Barras
                    ax.barh(y_pos - height/2, top_n['Margem Total Atual'],
                            height, label='Margem Atual', color='#444444')
                    bars_new = ax.barh(
                        y_pos + height/2, top_n['Margem Total Nova'], height, label='Nova Margem', color='#00e5ff')

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_n['Produto'], color='white')
                    ax.set_xlabel('Margem Total (€)', color='white')
                    ax.legend(facecolor='#1a1625', labelcolor='white')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.spines['bottom'].set_visible(True)
                    ax.spines['bottom'].set_color('white')

                    # Labels nas barras
                    for bar in bars_new:
                        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                                f' €{int(bar.get_width())}', va='center', color='white', fontsize=8)
                    st.pyplot(fig)

                # --- TABELA FINAL ---
                st.dataframe(
                    rdf.style.format({
                        'Margem Real': '{:.2f}€',
                        'Mg. Teórica At.': '{:.2f}€',
                        'Nova Margem': '{:.2f}€',
                        'Delta Preço': '{:+.2f}€',
                        'Ganho Est.': '{:.2f}€',
                        'Pr. Atual': '{:.2f}€',
                        'Pr. Novo': '{:.2f}€'
                    }).background_gradient(subset=['Ganho Est.'], cmap='Greens'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        # Colunas Escondidas (usadas só nos gráficos)
                        "Ganho Unit Delta": None, "Margem Total Atual": None, "Margem Total Nova": None,
                        # Formatação
                        "Pr. Atual": st.column_config.NumberColumn(f"{lbl_p} At.", format="%.2f €"),
                        "Pr. Novo": st.column_config.NumberColumn(f"{lbl_p} Novo", format="%.2f €"),
                        "Mg. Teórica At.": st.column_config.NumberColumn("Mg. Teórica (Atual)", help="O que ganharia se comprasse o produto atual hoje."),
                        "Sugestão": st.column_config.TextColumn("Sugestão de Troca"),
                    }
                )
                ui_help.show_glossary()
            else:
                st.info("Sem oportunidades de melhoria neste grupo.")
    elif isinstance(df, str):
        st.error(df)
else:
    st.info("A aguardar ficheiros...")
