import streamlit as st
import pandas as pd
import numpy as np
import os
import download_infarmed_data
import ui_help
import ui_style

# --- Configuração Inicial ---
st.set_page_config(page_title="Análise Genéricos ABC/XYZ", layout="wide")

# ==============================================================================
# 0. SISTEMA DE LOGIN (NOVO)
# ==============================================================================


def check_login():
    """Gere a autenticação simples."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Interface de Login
    st.markdown("### 🔐 Acesso Restrito")

    # Podes substituir este dicionário por st.secrets ou base de dados
    # Carrega os utilizadores diretamente do ficheiro secrets.toml
    try:
        USERS = st.secrets["users"]
    except FileNotFoundError:
        st.error("Ficheiro .streamlit/secrets.toml não encontrado!")
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
    st.stop()  # Pára a execução aqui se não estiver logado

# --- INÍCIO DA APLICAÇÃO (Apenas corre se autenticado) ---

master_path = "allPackages_python.xls"
if 'startup_check_done' not in st.session_state:
    if not os.path.exists(master_path):
        with st.spinner("⚠️ Base de dados em falta. A descarregar automaticamente do Infarmed..."):
            success = download_infarmed_data.download_infarmed_xls()
            if success:
                st.toast("Base de dados descarregada com sucesso!", icon="✅")
            else:
                st.error("Falha crítica no download automático.")
    st.session_state['startup_check_done'] = True

ui_style.init_session_state()
ui_style.apply_custom_style()

# --- Lógica de Negócio ---


def calcular_pva(pvp):
    """Cálculo do PVA conforme regras oficiais."""
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
    # ==============================================================================
    # 1. PROCESSAMENTO DE VENDAS
    # ==============================================================================
    try:
        df_sales = None
        try:
            file_sales.seek(0)
            df_sales = pd.read_csv(
                file_sales, sep='\t', encoding='latin-1', on_bad_lines='skip', dtype=str)
            if df_sales.shape[1] < 5:
                raise ValueError("Colunas insuficientes")
        except:
            try:
                file_sales.seek(0)
                df_sales = pd.read_csv(
                    file_sales, sep=';', encoding='latin-1', on_bad_lines='skip', dtype=str)
            except:
                return None, "Erro crítico ao ler ficheiro de Vendas."

        # Filtro Localização
        if 'DUV' in df_sales.columns and 'LOCALIZACAO' in df_sales.columns:
            df_sales['DUV_dt'] = pd.to_datetime(
                df_sales['DUV'], dayfirst=True, format='mixed', errors='coerce')
            valid_dates = df_sales.dropna(subset=['DUV_dt'])
            if not valid_dates.empty:
                max_date_idx = valid_dates['DUV_dt'].idxmax()
                target_loc = df_sales.loc[max_date_idx, 'LOCALIZACAO']
                df_sales = df_sales[df_sales['LOCALIZACAO']
                                    == target_loc].copy()

        col_grupo = None
        if 'GRUPOHOMOGENEO' in df_sales.columns:
            col_grupo = 'GRUPOHOMOGENEO'
        elif 'CODIGOHOMOGENEO' in df_sales.columns:
            col_grupo = 'CODIGOHOMOGENEO'

        if col_grupo:
            df_sales = df_sales[df_sales[col_grupo].notna() & (
                df_sales[col_grupo].str.strip() != '')]

        # Limpeza Numérica
        numeric_cols = ['PVP', 'PCU', 'PVP5']
        for c in numeric_cols:
            if c in df_sales.columns:
                df_sales[c] = df_sales[c].str.replace(',', '.', regex=False)
                df_sales[c] = pd.to_numeric(
                    df_sales[c], errors='coerce').fillna(0.0)

        v_cols = [f'V({i})' for i in range(1, 7)]
        for c in v_cols:
            if c in df_sales.columns:
                df_sales[c] = pd.to_numeric(
                    df_sales[c], errors='coerce').fillna(0)

        valid_v_cols = [c for c in v_cols if c in df_sales.columns]
        df_sales['VENDAS_6M'] = df_sales[valid_v_cols].sum(axis=1)

        if 'PVP' in df_sales.columns and 'PCU' in df_sales.columns:
            df_sales['MARGEM_REAL_UNIT'] = (
                (df_sales['PVP'] / 1.06) - df_sales['PCU']).round(2)
        else:
            df_sales['MARGEM_REAL_UNIT'] = 0.0

        df_sales['MARGEM_REAL_TOTAL'] = df_sales['MARGEM_REAL_UNIT'] * \
            df_sales['VENDAS_6M']
        df_sales['CPR'] = df_sales['CPR'].astype(
            str).str.split('.').str[0].str.strip()

        if col_grupo:
            df_sales['NOME_GRUPO_SALES'] = df_sales[col_grupo]

    except Exception as e:
        return None, f"Erro no Processamento de Vendas: {e}"

    # ==============================================================================
    # 2. PROCESSAMENTO MESTRE & DESCONTOS (ATUALIZADO PARA LABORATÓRIO)
    # ==============================================================================
    try:
        df_master = pd.read_excel(file_master)

        rename_map = {
            'Nº registo': 'N_REGISTO', 'Preço (PVP)': 'PVP',
            'Preço Utente': 'P_UTENTE', 'Preço Pensionistas': 'P_PENSIONISTA',
            'Nome do medicamento': 'Nome'
        }
        df_master.rename(columns=rename_map, inplace=True)
        df_master['N_REGISTO'] = pd.to_numeric(
            df_master['N_REGISTO'], errors='coerce').fillna(0).astype(int).astype(str)

        if 'CNPEM' in df_master.columns:
            df_master['CNPEM'] = pd.to_numeric(
                df_master['CNPEM'], errors='coerce')
            df_master = df_master.dropna(subset=['CNPEM'])

        if 'Comerc.' in df_master.columns:
            df_master = df_master[df_master['Comerc.'].astype(
                str).str.strip() != "Não comercializado"]

        for c in ['PVP', 'P_UTENTE', 'P_PENSIONISTA']:
            if c in df_master.columns:
                if df_master[c].dtype == object:
                    df_master[c] = df_master[c].astype(
                        str).str.replace(',', '.', regex=False)
                df_master[c] = pd.to_numeric(
                    df_master[c], errors='coerce').fillna(0.0)

        df_master['PVA'] = df_master['PVP'].apply(calcular_pva)

        # --- DESCONTOS COM LABORATÓRIO ---
        # Ler ficheiro de descontos
        df_desc = pd.read_excel(file_discounts)

        # Normalizar colunas (Procura LAB, Laboratorio, etc)
        df_desc.columns = [c.upper().strip() for c in df_desc.columns]

        # Mapear coluna de laboratório
        lab_col = next((c for c in df_desc.columns if 'LAB' in c), None)
        cnp_col = next((c for c in df_desc.columns if 'CNP' in c), 'CNP')
        desc_col = next((c for c in df_desc.columns if 'DESC' in c), 'DESC')

        # Limpeza
        df_desc[cnp_col] = pd.to_numeric(
            df_desc[cnp_col], errors='coerce').fillna(0).astype(int).astype(str)

        def clean_d(x):
            try:
                s = str(x).replace(',', '.')
                f = float(s)
                return f/100 if f > 1 else f
            except:
                return 0.0
        df_desc[desc_col] = df_desc[desc_col].apply(clean_d)

        if lab_col:
            df_desc[lab_col] = df_desc[lab_col].astype(str).str.strip()
        else:
            # Se não existir coluna Lab, cria uma "Geral"
            df_desc['LAB_DEFAULT'] = 'Geral'
            lab_col = 'LAB_DEFAULT'

        # Merge Descontos + Laboratório para o Mestre
        df_master = pd.merge(
            df_master,
            df_desc[[cnp_col, desc_col, lab_col]],
            left_on='N_REGISTO',
            right_on=cnp_col,
            how='left'
        )

        # Preencher vazios
        df_master['DESC_COMERCIAL'] = df_master[desc_col].fillna(0.0)
        df_master['LABORATORIO_DESC'] = df_master[lab_col].fillna(
            'Outros/Sem Acordo')

        # Calcular Margem Teórica
        df_master['PVA_LIQ'] = df_master['PVA'] * \
            (1 - df_master['DESC_COMERCIAL'])
        df_master['MARGEM_TEORICA'] = (
            df_master['PVP'] / 1.06) - df_master['PVA_LIQ']

    except Exception as e:
        return None, f"Erro no Processamento Mestre/Descontos: {e}"

    # ==============================================================================
    # 3. MERGE FINAL
    # ==============================================================================
    cols_sales = ['CPR', 'NOM', 'VENDAS_6M',
                  'MARGEM_REAL_UNIT', 'MARGEM_REAL_TOTAL']
    if 'NOME_GRUPO_SALES' in df_sales.columns:
        cols_sales.append('NOME_GRUPO_SALES')

    df_sales_clean = df_sales[cols_sales].copy()

    df_final = pd.merge(
        df_sales_clean,
        df_master,
        left_on='CPR',
        right_on='N_REGISTO',
        how='inner'
    )

    if 'CNPEM' not in df_final.columns:
        return None, "Erro Crítico: Coluna CNPEM perdida no merge."

    return df_final, df_master


def run_abc_analysis(df):
    if df.empty:
        return pd.DataFrame()

    group_col = 'NOME_GRUPO_SALES' if 'NOME_GRUPO_SALES' in df.columns else 'Substância Ativa/DCI'
    if group_col not in df.columns:
        group_col = 'CNPEM'

    grouped = df.groupby('CNPEM').agg({
        'MARGEM_REAL_TOTAL': 'sum',
        'VENDAS_6M': 'sum',
        'CPR': 'count',
        group_col: 'first'
    }).rename(columns={'CPR': 'NUM_PRODUTOS', group_col: 'NOME_GRUPO'})

    grouped = grouped.sort_values('MARGEM_REAL_TOTAL', ascending=False)

    total = grouped['MARGEM_REAL_TOTAL'].sum()
    if total == 0:
        total = 1

    grouped['PERC'] = grouped['MARGEM_REAL_TOTAL'].cumsum() / total

    def get_class(x):
        if x <= 0.80:
            return 'A'
        elif x <= 0.95:
            return 'B'
        return 'C'

    grouped['CLASSE_ABC'] = grouped['PERC'].apply(get_class)
    return grouped

# --- INTERFACE GRÁFICA PRINCIPAL ---


st.sidebar.markdown(
    f"👤 *Logado como: {st.session_state.get('username', 'Utilizador')}*")
if st.sidebar.button("Terminar Sessão"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.header("📁 Carregar Dados")
up_sales = st.sidebar.file_uploader(
    "Vendas (Infoprex .txt/.csv)", type=['txt', 'csv'])

# Auto Download
st.sidebar.markdown("### 2. Mestre (Infarmed)")
if st.sidebar.button("🔄 Atualizar Base de Dados"):
    with st.spinner("A descarregar do Infarmed..."):
        if download_infarmed_data.download_infarmed_xls():
            st.sidebar.success("Download com sucesso!")
            st.cache_data.clear()
        else:
            st.sidebar.error("Erro no download.")

has_master = os.path.exists(master_path)
if has_master:
    t = os.path.getmtime(master_path)
    dt = pd.to_datetime(t, unit='s')
    st.sidebar.caption(
        f"✅ Ficheiro disponível ({dt.strftime('%d/%m/%Y %H:%M')})")
else:
    st.sidebar.warning("⚠️ Ficheiro em falta.")

up_desc = st.sidebar.file_uploader("Descontos (.xlsx)", type=['xlsx'])

if up_sales and has_master and up_desc:
    # Processamento
    df, df_master = process_data(
        up_sales, master_path, up_desc, master_mtime=os.path.getmtime(master_path))

    if df is not None and not isinstance(df, str):
        # --- FILTRO DE LABORATÓRIOS (NOVO) ---
        # Obter lista de laboratórios disponíveis no ficheiro de descontos/mestre
     # --- FILTRO DE LABORATÓRIOS (COM BOTÕES DE CONTROLE) ---
        # Obter lista de laboratórios disponíveis
        all_labs = sorted(
            [str(x) for x in df_master['LABORATORIO_DESC'].unique() if pd.notna(x)])

        st.sidebar.divider()
        st.sidebar.markdown("### 🧪 Filtros de Substituição")

        # 1. Inicializar a lista no Session State se ainda não existir
        # Isto garante que o filtro começa com "Todos" selecionados por defeito
        if 'lab_filter_state' not in st.session_state:
            st.session_state['lab_filter_state'] = all_labs

        # 2. Botões de Ação Rápida
        # Usamos colunas para ficarem lado a lado
        col_btn1, col_btn2 = st.sidebar.columns(2)

        # Botão "Selecionar Todos"
        if col_btn1.button("✅ Todos", use_container_width=True):
            st.session_state['lab_filter_state'] = all_labs

        # Botão "Limpar" (Selecionar Nenhum)
        if col_btn2.button("❌ Limpar", use_container_width=True):
            st.session_state['lab_filter_state'] = []

        # 3. Multiselect vinculado ao Session State
        # O parâmetro 'key' liga este widget à variável que alterámos nos botões acima
        selected_labs = st.sidebar.multiselect(
            "Filtrar Laboratórios (Destino):",
            options=all_labs,
            key='lab_filter_state',
            help="As sugestões de troca serão limitadas a estes laboratórios."
        )

        # Se a lista estiver vazia (usuário limpou tudo), avisamos ou assumimos todos?
        # Lógica atual: Se vazio, não mostra nada (comportamento padrão do filtro isin)
        if not selected_labs:
            st.sidebar.warning(
                "⚠️ Selecione pelo menos um laboratório para ver sugestões.")

        abc = run_abc_analysis(df)
        class_a = abc[abc['CLASSE_ABC'] == 'A'].reset_index()
        class_a['LABEL'] = "#" + (class_a.index + 1).astype(str) + " - " + class_a['NOME_GRUPO'].astype(
            str) + " (" + class_a['MARGEM_REAL_TOTAL'].apply(lambda x: f"€{x:,.0f}") + ")"

        st.subheader("🎯 Grupos Prioritários (Classe A)")

        if not class_a.empty:
            sel_label = st.selectbox("Selecione Grupo:", class_a['LABEL'])
            sel_cnpem = class_a[class_a['LABEL']
                                == sel_label]['CNPEM'].values[0]

            # --- SIMULADOR ---
            st.divider()
            st.sidebar.markdown("### ⚙️ Simulação")
            regime = st.sidebar.radio("Regime:", ["Geral", "R Especial"])
            p_col = 'P_PENSIONISTA' if regime == "R Especial" else 'P_UTENTE'
            lbl_p = "Pr. Pen." if regime == "R Especial" else "Pr. Ut."

            tol = st.sidebar.number_input(
                f"Tolerância {lbl_p} (€)", 0.0, 5.0, 0.5, 0.1)

            # Dados
            products = df[df['CNPEM'] == sel_cnpem].copy(
            ).sort_values('VENDAS_6M', ascending=False)
            market = df_master[df_master['CNPEM'] == sel_cnpem].copy()

            results = []
            tot_gain = 0.0

            for _, row in products.iterrows():
                if row['VENDAS_6M'] <= 0:
                    continue

                curr_id = row['CPR']
                curr_margin_real = row['MARGEM_REAL_UNIT']
                curr_price_off = row[p_col]
                curr_margin_theo = row['MARGEM_TEORICA']

                # --- CANDIDATOS (COM FILTRO DE LAB) ---
                cands = market[
                    (market['MARGEM_TEORICA'] > curr_margin_theo) &
                    (market[p_col] <= (curr_price_off + tol)) &
                    (market['N_REGISTO'] != curr_id) &
                    # <--- FILTRO APLICADO
                    (market['LABORATORIO_DESC'].isin(selected_labs))
                ]

                if not cands.empty:
                    best = cands.sort_values(
                        'MARGEM_TEORICA', ascending=False).iloc[0]

                    gain_vs_stock = best['MARGEM_TEORICA'] - curr_margin_real
                    if gain_vs_stock > 0:
                        action = "Trocar Já 🔄"
                        prio = 1
                        val_gain = gain_vs_stock * row['VENDAS_6M']
                    else:
                        action = "Esgotar 📉"
                        prio = 2
                        val_gain = 0

                    if prio == 1:
                        tot_gain += val_gain

                    results.append({
                        'CNP': curr_id,
                        'Produto': row['NOM'],
                        'Vol': row['VENDAS_6M'],
                        'Pr. Atual': curr_price_off,
                        'CNP Novo': best['N_REGISTO'],
                        'Sugestão': best['Nome'],
                        # Mostra o Lab na tabela
                        # 'Lab. Sug.': best['LABORATORIO_DESC'],
                        'Pr. Novo': best[p_col],
                        'Margem Real': curr_margin_real,
                        'Margem Teórica': curr_margin_theo,
                        'Nova Margem': best['MARGEM_TEORICA'],
                        'Delta Preço': best[p_col] - curr_price_off,
                        'Ganho Est.': val_gain,
                        'Ação': action,
                        'Priority': prio,
                        'Ganho_Visual': val_gain if prio == 1 else (best['MARGEM_TEORICA'] - curr_margin_theo) * row['VENDAS_6M']
                    })

            if results:
                rdf = pd.DataFrame(results).sort_values(
                    ['Priority', 'Ganho Est.'], ascending=[True, False])

                st.metric("Potencial Ganho Imediato", f"€ {tot_gain:,.2f}")
                st.scatter_chart(rdf, x='Delta Preço', y='Ganho_Visual',
                                 color='Ação', size='Vol', height=400)

                st.dataframe(
                    rdf.style.format({
                        'Margem Real': '{:.2f}€', 'Margem Teórica': '{:.2f}€', 'Nova Margem': '{:.2f}€',
                        'Delta Preço': '{:+.2f}€', 'Ganho Est.': '{:.2f}€',
                        'Pr. Atual': '{:.2f}€', 'Pr. Novo': '{:.2f}€'
                    }).background_gradient(subset=['Ganho Est.'], cmap='Greens'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ganho_Visual": None,
                        "Priority": None,
                        "CNP": st.column_config.TextColumn("CNP"),
                        "CNP Novo": st.column_config.TextColumn("CNP Sug."),
                        # "Lab. Sug.": st.column_config.TextColumn("Laboratório"),
                        "Pr. Atual": st.column_config.NumberColumn(f"{lbl_p} At.", format="%.2f €"),
                        "Pr. Novo": st.column_config.NumberColumn(f"{lbl_p} Novo", format="%.2f €")
                    }
                )
                ui_help.show_glossary()
            else:
                st.info(
                    "Portefólio Otimizado (ou sem opções nos laboratórios selecionados).")
        else:
            st.warning("Sem dados Classe A.")
    elif isinstance(df, str):
        st.error(df)
    else:
        st.error("Erro desconhecido.")
else:
    st.info("A aguardar ficheiros...")
