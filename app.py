import base64
import hashlib
import os
from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
import streamlit as st

from job_classifier import Verbo, Patron, classify, VERBOS_VALIDOS


# =========================================================
# CONFIG
# =========================================================
from PIL import Image

logo = Image.open("assets/logo_elche.png")

col1, col2 = st.columns([1.2, 4])

with col1:
    st.image(logo, width=120)

with col2:
    st.markdown("""
    <div style="padding-top: 10px;">
        <h1 style="margin-bottom:0; color: rgb(11,16,29);">
            Clasificador de Puestos
        </h1>
        <p style="margin-top:4px; color:#4f5d69;">
            Evaluación funcional y propuesta orientativa de nivel de complemento de destino
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    :root {
        --azul-principal: rgb(0, 96, 137);
        --azul-oscuro: rgb(11, 16, 29);
        --azul-navy: #1b2a38;

        --gris-fondo: #aeb8c2;
        --gris-panel: #c7d0d8;
        --gris-panel-2: #bcc6cf;
        --gris-input: #d8e0e6;

        --gris-borde: #8998a4;
        --gris-texto: #18232c;
        --gris-texto-suave: #42515d;

        --success-bg: #5f8f72;
        --success-text: #ffffff;

        --info-bg: #5f7f99;
        --info-text: #ffffff;

        --warning-bg: #b98a43;
        --warning-text: #ffffff;

        --error-bg: #9e5d5d;
        --error-text: #ffffff;
    }

    /* Fondo general */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--gris-fondo) !important;
        color: var(--gris-texto) !important;
    }

    /* Contenido principal */
    .main .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Header superior */
    header[data-testid="stHeader"] {
        background: var(--azul-oscuro) !important;
    }

    div[data-testid="stToolbar"] {
        background: var(--azul-oscuro) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--azul-oscuro) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Títulos */
    h1, h2, h3 {
        color: var(--azul-oscuro) !important;
    }

    p, span, div, label {
        color: var(--gris-texto);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background: var(--gris-panel-2) !important;
        border: 1px solid var(--gris-borde) !important;
        border-radius: 10px 10px 0 0 !important;
        color: var(--gris-texto) !important;
        font-weight: 600 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--gris-panel) !important;
        color: var(--azul-principal) !important;
        border-bottom: 3px solid var(--azul-principal) !important;
    }

    /* Botones */
    .stButton > button,
    .stDownloadButton > button {
        background: var(--azul-principal) !important;
        color: white !important;
        border: 1px solid var(--azul-principal) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--azul-oscuro) !important;
        border-color: var(--azul-oscuro) !important;
        color: white !important;
    }

    /* FILE UPLOADER */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, var(--azul-navy), var(--azul-principal)) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        border: none !important;
    }

    div[data-testid="stFileUploader"] * {
        color: white !important;
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.10) !important;
        border: 1px dashed rgba(255,255,255,0.35) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFileUploader"] button {
        background: var(--azul-oscuro) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: #24384b !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
    }

    /* Inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        background: var(--gris-input) !important;
        color: var(--gris-texto) !important;
        border: 1px solid var(--gris-borde) !important;
        border-radius: 8px !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background: var(--gris-input) !important;
        color: var(--gris-texto) !important;
        border: 1px solid var(--gris-borde) !important;
        border-radius: 8px !important;
    }

    /* Alertas con más contraste */
    div[data-testid="stAlert"] {
        border: 1px solid var(--gris-borde) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stAlert"][kind="success"] {
        background: var(--success-bg) !important;
    }

    div[data-testid="stAlert"][kind="success"] * {
        color: var(--success-text) !important;
    }

    div[data-testid="stAlert"][kind="info"] {
        background: var(--info-bg) !important;
    }

    div[data-testid="stAlert"][kind="info"] * {
        color: var(--info-text) !important;
    }

    div[data-testid="stAlert"][kind="warning"] {
        background: var(--warning-bg) !important;
    }

    div[data-testid="stAlert"][kind="warning"] * {
        color: var(--warning-text) !important;
    }

    div[data-testid="stAlert"][kind="error"] {
        background: var(--error-bg) !important;
    }

    div[data-testid="stAlert"][kind="error"] * {
        color: var(--error-text) !important;
    }

    /* Dataframe / editor */
    div[data-testid="stDataFrame"] {
        background: var(--gris-panel) !important;
        border: 1px solid var(--gris-borde) !important;
        border-radius: 10px !important;
        padding: 0.25rem !important;
    }

    /* JSON */
    .stJson {
        background: var(--gris-panel) !important;
        border: 1px solid var(--gris-borde) !important;
        border-radius: 10px !important;
    }

    /* Expanders */
    details {
        background: var(--gris-panel) !important;
        border: 1px solid var(--gris-borde) !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.8rem !important;
    }

    /* Checkbox label general */
    div[data-testid="stCheckbox"] label {
        color: var(--azul-oscuro) !important;
        font-weight: 600 !important;
    }

    /* Texto específico del verbo */
    div[data-testid="stCheckbox"] p,
    div[data-testid="stCheckbox"] span {
        color: var(--azul-oscuro) !important;
        font-weight: 600 !important;
    }

    /* Separadores */
    hr {
        border: none !important;
        border-top: 1px solid var(--gris-borde) !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(
    "Carga un archivo de configuración al iniciar la sesión. "
    "Después podrás clasificar, editar variables, exportar una nueva configuración "
    "y generar un informe técnico en HTML."
)


# =========================================================
# UTILIDADES GENERALES
# =========================================================
def normaliza_texto(texto):
    if pd.isna(texto):
        return ""
    return (
        str(texto)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )


def normaliza_serie_texto(serie):
    return serie.astype(str).str.strip().str.lower()


def normaliza_verbo_id(valor):
    if pd.isna(valor):
        return ""

    txt = str(valor).strip()

    try:
        num = float(txt)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return txt.lower()


def mostrar_verbo(verbo):
    verbo = str(verbo).strip()
    return verbo.capitalize() if verbo else verbo


def es_columna_patron(col):
    c = str(col).strip().lower()
    if c == "":
        return False

    excluidas = {
        "verbo_id",
        "verbo",
        "grupo_verbo",
        "definicion",
        "categoria",
        "dfi numero apariciones",
        "dfi nº apariciones",
        "cdmedio(i)=dfi1∑cdj",
        "rango=|cdmax-cdmin|",
    }
    if c in excluidas:
        return False

    try:
        float(c)
        return True
    except ValueError:
        return False


def etiqueta_grupo(grupo):
    etiquetas = {
        "operacion": "Operación",
        "mando": "Mando",
        "tecnicos": "Técnicos",
        "direccion": "Dirección",
    }
    return etiquetas.get(grupo, str(grupo).capitalize())


def file_to_bytes(uploaded_file):
    return uploaded_file.getvalue()


def hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def normalizar_nombres_columnas(df):
    nuevas_columnas = []
    for col in df.columns:
        if isinstance(col, str):
            col_norm = normaliza_texto(col)
            if col_norm.startswith("unnamed"):
                nuevas_columnas.append("")
            else:
                nuevas_columnas.append(col_norm)
        else:
            nuevas_columnas.append(col)
    df.columns = nuevas_columnas
    return df


def limpiar_checkboxes_clasificacion():
    claves_a_borrar = [
        k for k in st.session_state.keys()
        if isinstance(k, str) and k.startswith("chk_")
    ]
    for k in claves_a_borrar:
        del st.session_state[k]


# =========================================================
# CARGA Y NORMALIZACIÓN DE HOJAS
# =========================================================
def cargar_hoja_verbs_desde_df(df):
    df = df.dropna(axis=1, how="all").copy()
    df = normalizar_nombres_columnas(df)

    df = df.loc[:, [c for c in df.columns if c != ""]].copy()
    df = df.dropna(axis=0, how="all").copy()

    if "verbo_id" in df.columns:
        df["verbo_id"] = df["verbo_id"].apply(normaliza_verbo_id)

    if "verbo" in df.columns:
        df["verbo"] = normaliza_serie_texto(df["verbo"])

    if "grupo_verbo" in df.columns:
        df["grupo_verbo"] = df["grupo_verbo"].fillna("").astype(str).str.strip()

    if "definicion" in df.columns:
        df["definicion"] = df["definicion"].fillna("").astype(str).str.strip()

    if "categoria" in df.columns:
        df["categoria"] = normaliza_serie_texto(df["categoria"])

    if "verbo_id" in df.columns:
        df = df[df["verbo_id"].notna()]
        df = df[df["verbo_id"].astype(str).str.strip() != ""]

    if "verbo" in df.columns:
        df = df[df["verbo"].notna()]
        df = df[df["verbo"].astype(str).str.strip() != ""]

    pattern_cols = [c for c in df.columns if es_columna_patron(c)]
    for col in pattern_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def cargar_hoja_compat_desde_df(df):
    df = df.dropna(axis=1, how="all").copy()

    columnas_nuevas = []
    for i, col in enumerate(df.columns):
        if i == 0:
            columnas_nuevas.append("verbo_id")
        else:
            columnas_nuevas.append(normaliza_texto(col))

    df.columns = columnas_nuevas

    if "operación" in df.columns:
        df = df.rename(columns={"operación": "operacion"})

    df = df.loc[:, [c for c in df.columns if c != ""]].copy()
    df = df.dropna(axis=0, how="all").copy()

    if "verbo_id" in df.columns:
        df["verbo_id"] = df["verbo_id"].apply(normaliza_verbo_id)
        df = df[df["verbo_id"].notna()]
        df = df[df["verbo_id"].astype(str).str.strip() != ""]

    for col in ["operacion", "mando", "tecnicos", "direccion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def cargar_hoja_verbs(uploaded_file_or_bytes):
    intentos = [3, 0]
    ultimo_error = None

    for header in intentos:
        try:
            df = pd.read_excel(
                BytesIO(uploaded_file_or_bytes) if isinstance(uploaded_file_or_bytes, bytes) else uploaded_file_or_bytes,
                sheet_name="verbs_patterns",
                header=header,
            )
            df = cargar_hoja_verbs_desde_df(df)

            columnas = set(df.columns)
            required = {"verbo_id", "verbo", "grupo_verbo", "definicion", "categoria"}
            if required.issubset(columnas):
                return df
        except Exception as e:
            ultimo_error = e

    raise ValueError(
        "No se pudo leer correctamente la hoja 'verbs_patterns'. "
        f"Último error: {ultimo_error}"
    )


def cargar_hoja_compat(uploaded_file_or_bytes):
    intentos = [2, 0]
    ultimo_error = None

    for header in intentos:
        try:
            df = pd.read_excel(
                BytesIO(uploaded_file_or_bytes) if isinstance(uploaded_file_or_bytes, bytes) else uploaded_file_or_bytes,
                sheet_name="compatibility_verbs",
                header=header,
            )

            df = df.dropna(axis=1, how="all").copy()

            columnas_nuevas = []
            for i, col in enumerate(df.columns):
                if i == 0:
                    columnas_nuevas.append("verbo_id")
                else:
                    columnas_nuevas.append(normaliza_texto(col))

            df.columns = columnas_nuevas

            if "operación" in df.columns:
                df = df.rename(columns={"operación": "operacion"})

            df = df.dropna(axis=0, how="all").copy()

            df["verbo_id"] = df["verbo_id"].apply(normaliza_verbo_id)
            df = df[df["verbo_id"].astype(str).str.strip() != ""]

            for col in ["operacion", "mando", "tecnicos", "direccion"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

            required = {"verbo_id", "operacion", "mando", "tecnicos", "direccion"}
            if required.issubset(set(df.columns)):
                return df

        except Exception as e:
            ultimo_error = e

    raise ValueError(
        "No se pudo leer correctamente la hoja 'compatibility_verbs'. "
        f"Último error: {ultimo_error}"
    )


# =========================================================
# VALIDACIONES
# =========================================================
def validar_columnas(df_verbs, df_comp):
    required_verbs = {"verbo_id", "verbo", "grupo_verbo", "definicion", "categoria"}
    required_comp = {"verbo_id", "operacion", "mando", "tecnicos", "direccion"}

    missing_verbs = required_verbs - set(df_verbs.columns)
    missing_comp = required_comp - set(df_comp.columns)

    if missing_verbs:
        raise ValueError(f"Faltan columnas en 'verbs_patterns': {sorted(missing_verbs)}")

    if missing_comp:
        raise ValueError(f"Faltan columnas en 'compatibility_verbs': {sorted(missing_comp)}")


def validar_verbos_excel(df_verbs):
    verbos_excel = set(df_verbs["verbo"].dropna().unique())
    verbos_modelo = set(VERBOS_VALIDOS)

    no_validos = sorted(verbos_excel - verbos_modelo)
    faltantes = sorted(verbos_modelo - verbos_excel)

    if no_validos:
        raise ValueError(f"Hay verbos en el Excel que no existen en el motor: {no_validos}")

    return faltantes


def validar_valores_gestion(df_verbs, df_comp):
    validar_columnas(df_verbs, df_comp)
    faltantes = validar_verbos_excel(df_verbs)

    categorias_validas = {"nuclear", "relevante", "apoyo", "accesorio"}
    categorias_detectadas = set(df_verbs["categoria"].dropna().astype(str).str.strip().str.lower())
    categorias_invalidas = sorted(categorias_detectadas - categorias_validas)
    if categorias_invalidas:
        raise ValueError(
            f"Hay categorías inválidas en verbs_patterns: {categorias_invalidas}. "
            f"Válidas: {sorted(categorias_validas)}"
        )

    duplicados_verbo_id_verbs = (
        df_verbs["verbo_id"].astype(str).str.strip().duplicated().any()
    )
    if duplicados_verbo_id_verbs:
        raise ValueError("Hay verbo_id duplicados en 'verbs_patterns'.")

    duplicados_verbo_id_comp = (
        df_comp["verbo_id"].astype(str).str.strip().duplicated().any()
    )
    if duplicados_verbo_id_comp:
        raise ValueError("Hay verbo_id duplicados en 'compatibility_verbs'.")

    ids_verbs = set(df_verbs["verbo_id"].astype(str).str.strip())
    ids_comp = set(df_comp["verbo_id"].astype(str).str.strip())

    solo_verbs = sorted(ids_verbs - ids_comp)
    solo_comp = sorted(ids_comp - ids_verbs)

    if solo_verbs or solo_comp:
        msg = []
        if solo_verbs:
            msg.append(f"IDs presentes solo en verbs_patterns: {solo_verbs[:10]}")
        if solo_comp:
            msg.append(f"IDs presentes solo en compatibility_verbs: {solo_comp[:10]}")
        raise ValueError("Inconsistencia entre hojas. " + " | ".join(msg))

    pattern_cols = obtener_columnas_patron(df_verbs)
    if not pattern_cols:
        raise ValueError("No se han detectado columnas de patrón válidas.")

    return faltantes


# =========================================================
# COMPATIBILIDAD FUNCIONAL
# =========================================================
def obtener_grupos_compatibles(verbo_id, df_comp):
    verbo_id_norm = normaliza_verbo_id(verbo_id)
    comp_row = df_comp[df_comp["verbo_id"] == verbo_id_norm]

    if comp_row.empty:
        return []

    grupos = []
    for g in ["operacion", "mando", "tecnicos", "direccion"]:
        val = comp_row.iloc[0][g]
        if pd.notna(val) and int(val) == 1:
            grupos.append(g)

    return grupos


def analizar_incompatibilidades(selected_rows, df_comp):
    detalle = []

    for row in selected_rows:
        verbo_id = normaliza_verbo_id(row["verbo_id"])
        verbo = str(row["verbo"]).strip()
        grupos = obtener_grupos_compatibles(verbo_id, df_comp)

        detalle.append({
            "verbo_id": verbo_id,
            "verbo": verbo,
            "grupos": grupos
        })

    verbos_operacion = [x for x in detalle if "operacion" in x["grupos"]]
    verbos_direccion = [x for x in detalle if "direccion" in x["grupos"]]

    hay_conflicto = len(verbos_operacion) > 0 and len(verbos_direccion) > 0

    return {
        "hay_conflicto": hay_conflicto,
        "verbos_operacion": verbos_operacion,
        "verbos_direccion": verbos_direccion,
        "detalle": detalle,
    }


# =========================================================
# PATRONES
# =========================================================
def obtener_columnas_patron(df_verbs):
    return [c for c in df_verbs.columns if es_columna_patron(c)]


def construir_patrones(df_verbs, pattern_cols):
    patrones = []

    for i, col in enumerate(pattern_cols, start=1):
        verbos_patron = []

        for _, row in df_verbs.iterrows():
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val) and int(val) == 1:
                verbos_patron.append(
                    Verbo(
                        nombre=str(row["verbo"]).strip().lower(),
                        categoria=str(row["categoria"]).strip().lower()
                    )
                )

        if verbos_patron:
            cd = float(str(col).strip())
            patron_id = f"p{i:02d}_{str(col).replace('.', '_')}"
            patrones.append(
                Patron(
                    id=patron_id,
                    nombre=patron_id,
                    cd=cd,
                    verbos=verbos_patron
                )
            )

    return patrones


# =========================================================
# EXPORTACIÓN DE CONFIGURACIÓN
# =========================================================
def exportar_excel_configuracion(df_verbs, df_comp):
    output = BytesIO()

    verbs_export = df_verbs.copy()
    comp_export = df_comp.copy()

    verbs_export.columns = [str(c) for c in verbs_export.columns]
    comp_export.columns = [str(c) for c in comp_export.columns]

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        verbs_export.to_excel(writer, sheet_name="verbs_patterns", index=False)
        comp_export.to_excel(writer, sheet_name="compatibility_verbs", index=False)

    output.seek(0)
    return output.getvalue()


# =========================================================
# INFORME TÉCNICO
# =========================================================
def normaliza_grupo_verbo(grupo):
    g = str(grupo).strip().lower()

    equivalencias = {
        "mando": "Mando",
        "resp. profes.": "Responsabilidad profesional",
        "resp. profes": "Responsabilidad profesional",
        "responsabilidad profesional": "Responsabilidad profesional",
        "espec. tecnica": "Especialización técnica",
        "espec. técnica": "Especialización técnica",
        "especializacion tecnica": "Especialización técnica",
        "especialización técnica": "Especialización técnica",
        "dif. tecnica": "Dificultad técnica",
        "dif. técnica": "Dificultad técnica",
        "dificultad tecnica": "Dificultad técnica",
        "dificultad técnica": "Dificultad técnica",
    }

    return equivalencias.get(g, str(grupo).strip())


def contar_grupos_por_verbos(nombres_verbos, df_verbs):
    conteo = {
        "Mando": 0,
        "Responsabilidad profesional": 0,
        "Especialización técnica": 0,
        "Dificultad técnica": 0,
    }

    if not nombres_verbos:
        return conteo

    verbos_norm = {str(v).strip().lower() for v in nombres_verbos}

    df_aux = df_verbs.copy()
    df_aux["verbo"] = df_aux["verbo"].astype(str).str.strip().str.lower()
    df_aux["grupo_verbo_norm"] = df_aux["grupo_verbo"].apply(normaliza_grupo_verbo)

    df_filtrado = df_aux[df_aux["verbo"].isin(verbos_norm)]

    for _, row in df_filtrado.iterrows():
        grupo = row["grupo_verbo_norm"]
        if grupo in conteo:
            conteo[grupo] += 1

    return conteo


def generar_frase_factores_cd(verbos_usuario, verbos_top1, df_verbs):
    verbos_totales = list({*verbos_usuario, *verbos_top1})
    conteo = contar_grupos_por_verbos(verbos_totales, df_verbs)

    grupos_ordenados = sorted(conteo.items(), key=lambda x: x[1], reverse=True)
    grupos_presentes = [(g, n) for g, n in grupos_ordenados if n > 0]

    if not grupos_presentes:
        return (
            "un perfil funcional sin predominio claro de factores, "
            "por lo que se aconseja una valoración complementaria del contenido del puesto"
        )

    principal, _ = grupos_presentes[0]
    secundarios = [g for g, n in grupos_presentes[1:] if n > 0]

    if principal == "Mando":
        base = "un componente predominante de mando"
    elif principal == "Responsabilidad profesional":
        base = "un componente predominante de responsabilidad profesional"
    elif principal == "Especialización técnica":
        base = "un componente predominante de especialización técnica"
    elif principal == "Dificultad técnica":
        base = "un componente predominante de dificultad técnica"
    else:
        base = f"un componente predominante de {principal.lower()}"

    if not secundarios:
        frase = base
    elif len(secundarios) == 1:
        frase = f"{base}, con presencia adicional de {secundarios[0].lower()}"
    else:
        frase = (
            f"{base}, con presencia adicional de "
            f"{', '.join(s.lower() for s in secundarios[:-1])} "
            f"y {secundarios[-1].lower()}"
        )

    interpretacion = {
        "Mando": (
            "compatible con puestos en los que concurren funciones de dirección, "
            "coordinación, organización o supervisión"
        ),
        "Responsabilidad profesional": (
            "compatible con puestos en los que adquieren especial relevancia "
            "la toma de decisiones, la definición funcional o la asunción de responsabilidad"
        ),
        "Especialización técnica": (
            "compatible con puestos en los que predomina el análisis, estudio, "
            "evaluación o desarrollo técnico"
        ),
        "Dificultad técnica": (
            "compatible con puestos en los que destacan exigencias de ejecución, "
            "apoyo funcional o complejidad operativa"
        ),
    }

    cierre = interpretacion.get(
        principal,
        "compatible con el encuadramiento funcional resultante"
    )

    return f"{frase}, {cierre}"


def img_to_base64(path_imagen):
    if not os.path.exists(path_imagen):
        return None
    with open(path_imagen, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def formatea_lista_html(items):
    if not items:
        return "No se aprecian elementos destacables"
    return ", ".join(escape(str(x)) for x in items)


def calcular_tendencia(cd_pred, cd_cont):
    diferencia = float(cd_cont) - float(cd_pred)

    if diferencia > 0:
        return "ascendente"
    elif diferencia < 0:
        return "descendente"
    return "neutra"


def generar_html_informe(
    result,
    selected_rows,
    df_verbs,
    configuracion_nombre,
    puesto_referencia="No indicado",
):
    top1 = result["top_k"][0] if result.get("top_k") else {
        "nombre": "No disponible",
        "similitud": "N/D",
        "cd": "N/D",
    }

    explicacion = result.get("explicacion", {})
    verbos_comunes = explicacion.get("verbos_comunes", [])
    solo_usuario = explicacion.get("solo_en_usuario", [])
    solo_top1 = explicacion.get("solo_en_top1", [])

    verbos_usuario = [
        str(row["verbo"]).strip().lower()
        for row in selected_rows
    ]

    verbos_top1 = list(set(verbos_comunes + solo_top1))

    factores_cd = generar_frase_factores_cd(
        verbos_usuario=verbos_usuario,
        verbos_top1=verbos_top1,
        df_verbs=df_verbs,
    )

    cd_pred = result.get("cd_pred", "N/D")
    cd_cont = result.get("cd_cont", "N/D")
    tendencia = calcular_tendencia(cd_pred, cd_cont)
    fecha_informe = datetime.now().strftime("%d/%m/%Y %H:%M")

    logo_base64 = img_to_base64("LOGO-HOR-CAS AZUL.png")

    logo_html = ""
    if logo_base64:
        logo_html = f"""
        <div class="logo-wrap">
            <img src="data:image/png;base64,{logo_base64}" alt="Ayuntamiento de Elche" class="logo" />
        </div>
        """
    else:
        logo_html = """
        <div class="logo-texto-fallback">
            Ayuntamiento de Elche
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Informe técnico CD</title>
  <style>
    :root {{
      --azul-principal: rgb(0, 96, 137);
      --azul-oscuro: rgb(11, 16, 29);
      --gris-fondo: #eef1f3;
      --gris-suave: #f7f8fa;
      --gris-borde: #cfd6dc;
      --gris-texto: #2f3941;
      --gris-meta: #5f6b75;
      --blanco: #ffffff;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 0;
      background: var(--gris-fondo);
      color: var(--gris-texto);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.55;
    }}

    .page {{
      max-width: 960px;
      margin: 0 auto;
      background: var(--blanco);
      min-height: 100vh;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }}

    .topbar {{
      height: 18px;
      background: var(--azul-oscuro);
    }}

    .hero {{
      background: var(--gris-fondo);
      padding: 34px 48px 26px 48px;
      border-bottom: 1px solid var(--gris-borde);
    }}

    .hero-inner {{
      display: flex;
      align-items: flex-start;
      gap: 24px;
    }}

    .hero-accent {{
      width: 10px;
      min-width: 10px;
      background: var(--azul-principal);
      border-radius: 2px;
      min-height: 128px;
    }}

    .hero-content {{
      flex: 1;
    }}

    .logo-wrap {{
      margin-bottom: 18px;
    }}

    .logo {{
      max-width: 360px;
      height: auto;
      display: block;
    }}

    .logo-texto-fallback {{
      font-size: 22px;
      font-weight: bold;
      color: var(--azul-principal);
      margin-bottom: 18px;
    }}

    h1 {{
      margin: 0 0 14px 0;
      font-size: 28px;
      line-height: 1.2;
      color: var(--azul-oscuro);
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}

    .meta {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 18px;
      font-size: 13px;
      color: var(--gris-meta);
      margin-top: 8px;
    }}

    .meta div {{
      padding: 4px 0;
    }}

    .content {{
      padding: 36px 48px 50px 48px;
    }}

    h2 {{
      font-size: 17px;
      color: var(--azul-principal);
      margin-top: 28px;
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 2px solid var(--gris-borde);
    }}

    p {{
      margin: 10px 0;
      text-align: justify;
    }}

    .bloque-datos {{
      background: var(--gris-suave);
      border: 1px solid var(--gris-borde);
      border-left: 6px solid var(--azul-principal);
      padding: 16px 18px;
      margin: 16px 0 18px 0;
      border-radius: 4px;
    }}

    .dato {{
      margin: 6px 0;
      font-size: 14px;
    }}

    .destacado {{
      font-weight: bold;
      color: var(--azul-oscuro);
    }}

    .firma-final {{
      margin-top: 34px;
      padding-top: 16px;
      border-top: 1px solid var(--gris-borde);
      font-size: 13px;
      color: var(--gris-meta);
    }}

    .footer-band {{
      margin-top: 40px;
      background: var(--azul-oscuro);
      color: var(--blanco);
      padding: 14px 48px;
      font-size: 12px;
      letter-spacing: 0.2px;
    }}

    @media print {{
      body {{
        background: white;
      }}
      .page {{
        box-shadow: none;
        max-width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="topbar"></div>

    <section class="hero">
      <div class="hero-inner">
        <div class="hero-accent"></div>
        <div class="hero-content">
          {logo_html}

          <h1>Informe técnico sobre determinación orientativa del nivel de complemento de destino</h1>

          <div class="meta">
            <div><strong>Fecha:</strong> {escape(fecha_informe)}</div>
            <div><strong>Puesto / referencia:</strong> {escape(str(puesto_referencia))}</div>
            <div><strong>Configuración utilizada:</strong> {escape(str(configuracion_nombre))}</div>
            <div><strong>Órgano emisor:</strong> Planificación de RR. HH.</div>
          </div>
        </div>
      </div>
    </section>

    <section class="content">
      <h2>Objeto</h2>
      <p>
        El presente informe tiene por objeto exponer el resultado de la evaluación funcional del puesto analizado, a efectos de formular una propuesta orientativa de nivel de complemento de destino, a partir del sistema de clasificación basado en verbos funcionales y en su comparación con patrones de referencia previamente definidos.
      </p>

      <h2>Marco técnico de referencia</h2>
      <p>
        La metodología utilizada parte del diccionario de funciones, que constituye la referencia para la identificación del nivel de especialización, responsabilidad, competencia, mando y complejidad funcional asignable a cada agrupación funcional.
      </p>

      <h2>Metodología aplicada</h2>
      <p>
        La herramienta compara la selección de verbos funcionales atribuida al puesto objeto de análisis con la matriz de patrones de referencia disponible en la configuración activa. Como resultado de dicha comparación, el sistema obtiene un CD predicho, un CD continuo y un conjunto de patrones con mayor similitud, incorporando además una explicación de las coincidencias y diferencias entre la selección efectuada y el patrón más próximo. Este resultado se interpreta de forma coherente con la gradación cualitativa de ámbitos funcionales prevista en el documento técnico de referencia.
      </p>

      <h2>Resultado obtenido</h2>
      <div class="bloque-datos">
        <div class="dato"><span class="destacado">CD predicho:</span> {escape(str(cd_pred))}</div>
        <div class="dato"><span class="destacado">CD continuo:</span> {escape(str(cd_cont))}</div>
        <div class="dato"><span class="destacado">Tendencia:</span> {escape(tendencia)}</div>
        <div class="dato"><span class="destacado">Patrón con mayor similitud:</span> {escape(str(top1["nombre"]))}</div>
        <div class="dato"><span class="destacado">Similitud principal:</span> {escape(str(top1["similitud"]))}</div>
      </div>

      <h2>Fundamentación técnica</h2>
      <p>
        La propuesta se apoya, principalmente, en la coincidencia funcional observada entre la selección analizada y el patrón de referencia de mayor proximidad, destacando los siguientes verbos comunes:
        <strong>{formatea_lista_html(verbos_comunes)}</strong>.
      </p>

      <p>
        Asimismo, se observan como elementos diferenciales los verbos presentes únicamente en la selección del puesto:
        <strong>{formatea_lista_html(solo_usuario)}</strong>,
        así como los verbos presentes únicamente en el patrón de referencia:
        <strong>{formatea_lista_html(solo_top1)}</strong>.
      </p>

      <p>
        Desde la perspectiva de los factores del complemento de destino, el conjunto funcional analizado presenta
        <strong>{escape(factores_cd)}</strong>,
        compatible con un encuadramiento en la banda correspondiente al nivel resultante, sin perjuicio de la valoración conjunta que proceda realizar con el resto de elementos organizativos del puesto.
      </p>

      <h2>Conclusión</h2>
      <p>
        En atención a cuanto antecede, se formula como resultado orientativo la asignación de un nivel de complemento de destino <strong>CD {escape(str(cd_pred))}</strong>, apreciándose una tendencia <strong>{escape(tendencia)}</strong> en función del CD continuo calculado.
      </p>

      <p>
        El presente informe tiene carácter técnico y constituye un elemento de apoyo a la valoración funcional del puesto dentro del marco normativo y organizativo aplicable.
      </p>

      <div class="firma-final">
        Documento generado automáticamente a partir de la configuración activa de la sesión y de los resultados de clasificación obtenidos.
      </div>
    </section>

    <div class="footer-band">
      Excmo. Ayuntamiento de Elche · Planificación de Recursos Humanos
    </div>
  </div>
</body>
</html>
"""
    return html


# =========================================================
# CARGA OBLIGATORIA DE CONFIGURACIÓN
# =========================================================
st.subheader("1) Carga de configuración")
uploaded_file = st.file_uploader(
    "📂 Cargar archivo de configuración (.xlsx)",
    type=["xlsx"],
    help=(
        "Este archivo define los verbos, compatibilidades y patrones de la sesión actual. "
        "Puede ser el Excel original o uno exportado por esta misma aplicación."
    ),
)

if not uploaded_file:
    st.info(
        "Debes cargar un archivo de configuración para comenzar. "
        "Hasta entonces, la app se queda en modo contemplativo."
    )
    st.stop()

try:
    uploaded_bytes = file_to_bytes(uploaded_file)
    uploaded_hash = hash_bytes(uploaded_bytes)

    if st.session_state.get("config_hash") != uploaded_hash:
        df_verbs = cargar_hoja_verbs(uploaded_bytes)
        df_comp = cargar_hoja_compat(uploaded_bytes)

        validar_columnas(df_verbs, df_comp)
        faltantes = validar_verbos_excel(df_verbs)

        st.session_state.df_verbs = df_verbs.copy()
        st.session_state.df_comp = df_comp.copy()
        st.session_state.df_verbs_base = df_verbs.copy()
        st.session_state.df_comp_base = df_comp.copy()
        st.session_state.config_hash = uploaded_hash
        st.session_state.config_name = uploaded_file.name
        st.session_state.verbos_faltantes = faltantes

        limpiar_checkboxes_clasificacion()

    st.success(f"Configuración activa cargada: {st.session_state.get('config_name', uploaded_file.name)}")

except Exception as e:
    st.error(f"Error al cargar la configuración: {e}")
    st.exception(e)
    st.stop()


def obtener_puesto_orientativo_por_cd(cd):
    """
    Devuelve una denominación orientativa de puesto a partir del CD predicho.
    """
    try:
        cd = int(float(cd))
    except Exception:
        return "No indicado"

    mapa = {
        30: "Persona habilitada nacional / Comisario principal jefe",
        29: "Dirección de área",
        28: "Jefatura de servicio / Comisario principal / Letrada asesoría jurídica / Coordinación",
        27: "Jefatura de sección / Sección técnica / Dirección CRIS / Dirección museos",
        26: "Jefatura de sección / Sección técnica / Comisario PL / Jefa-Jefe de gestión / Dirección deportiva / Dirección-Coordinación de EEII",
        25: "Puesto base de técnico superior",
        24: "Puesto base de técnico medio / Intendente PL",
        23: "Puesto de nivel intermedio (requiere concreción organizativa)",
        22: "Jefatura de departamento",
        21: "Jefatura de negociado / Unidad / Coordinación / Encargado / Oficial PL",
        20: "Puesto base de administrativo/a / Técnico especialista / Técnico auxiliar / Agente PL",
        19: "Puesto de operación-administración o técnico auxiliar (requiere concreción organizativa)",
        18: "Puesto de grupo C2",
        17: "Puesto base de auxiliar A.G. / Oficial (oficios) / Agente movilidad",
        16: "Personal ejecutivo",
        15: "Personal ejecutivo",
        14: "Puesto base de ordenanza / Portero grupo escolar / Ayudante de oficios",
    }

    return mapa.get(cd, f"CD {cd} – denominación orientativa no parametrizada")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("ℹ️ Sesión")
st.sidebar.write(f"**Archivo activo:** {st.session_state.get('config_name', '-')}")
st.sidebar.write("**Reglas actuales:**")
st.sidebar.write("• No combinar verbos incompatibles entre Operación y Dirección")
st.sidebar.write("• Verbos multigrupo son válidos")
st.sidebar.write("• La clasificación usa siempre la configuración activa de esta sesión")

modo_tecnico = st.sidebar.checkbox("🔧 Modo técnico")

if st.session_state.get("verbos_faltantes"):
    st.sidebar.warning(
        "Hay verbos del motor que no aparecen en la configuración cargada: "
        + ", ".join(st.session_state["verbos_faltantes"])
    )


# =========================================================
# PESTAÑAS
# =========================================================
tab1, tab2 = st.tabs(["🧠 Clasificación", "⚙️ Gestión de variables"])


# =========================================================
# TAB 1 — CLASIFICACIÓN
# =========================================================
with tab1:
    try:
        df_verbs = st.session_state.df_verbs.copy()
        df_comp = st.session_state.df_comp.copy()

        validar_valores_gestion(df_verbs, df_comp)
        pattern_cols = obtener_columnas_patron(df_verbs)

        if modo_tecnico:
            with st.expander("Diagnóstico de carga"):
                st.write("Columnas verbs_patterns:", list(df_verbs.columns))
                st.write("Columnas compatibility_verbs:", list(df_comp.columns))
                st.write("Primeras filas verbs_patterns:")
                st.dataframe(df_verbs.head(), use_container_width=True)
                st.write("Primeras filas compatibility_verbs:")
                st.dataframe(df_comp.head(), use_container_width=True)

            with st.expander("Diagnóstico de compatibilidad"):
                muestra = df_verbs[["verbo_id", "verbo"]].head(10).copy()
                muestra["grupos"] = muestra["verbo_id"].apply(
                    lambda x: ", ".join(obtener_grupos_compatibles(x, df_comp))
                )
                st.dataframe(muestra, use_container_width=True)

        st.subheader("Selecciona verbos")
        selected = []

        for idx, row in df_verbs.iterrows():
            nombre_verbo = mostrar_verbo(row["verbo"])
            id_verbo = normaliza_verbo_id(row["verbo_id"])
            definicion = str(row["definicion"]).strip() if pd.notna(row["definicion"]) else ""
            grupos = obtener_grupos_compatibles(id_verbo, df_comp)

            col1, col2 = st.columns([1, 3])

            with col1:
                checked = st.checkbox(nombre_verbo, key=f"chk_{id_verbo}_{idx}")

            with col2:
                if definicion:
                    st.caption(definicion)

                if grupos:
                    grupos_txt = ", ".join(etiqueta_grupo(g) for g in grupos)
                    st.write(f"**Compatible con:** {grupos_txt}")

            if checked:
                selected.append(row)

        if st.button("🔍 Clasificar"):
            if not selected:
                st.warning("Selecciona al menos un verbo.")
                st.stop()

            analisis = analizar_incompatibilidades(selected, df_comp)

            if analisis["hay_conflicto"]:
                verbos_op = sorted({mostrar_verbo(x["verbo"]) for x in analisis["verbos_operacion"]})
                verbos_dir = sorted({mostrar_verbo(x["verbo"]) for x in analisis["verbos_direccion"]})

                msg = (
                    "❌ Incompatibilidad detectada.\n\n"
                    f"**Verbos de Operación:** {', '.join(verbos_op)}\n\n"
                    f"**Verbos de Dirección:** {', '.join(verbos_dir)}\n\n"
                    "No pueden combinarse verbos de Operación con verbos de Dirección "
                    "en una misma selección."
                )
                st.error(msg)
                st.stop()

            verbos_usuario = [
                Verbo(
                    nombre=str(row["verbo"]).strip().lower(),
                    categoria=str(row["categoria"]).strip().lower()
                )
                for row in selected
            ]

            patrones = construir_patrones(df_verbs, pattern_cols)

            if not patrones:
                st.error("No se han construido patrones válidos a partir de la configuración.")
                st.stop()

            result = classify(verbos_usuario, patrones)

            st.session_state.last_result = result
            st.session_state.last_selected = selected

            st.success(f"CD Predicho: {result['cd_pred']}")
            st.info(f"CD Continuo: {result['cd_cont']}")

            if result.get("top_k"):
                top1 = result["top_k"][0]
                st.markdown(
                    f"**Patrón más similar:** {top1['nombre']} "
                    f"(CD {top1['cd']}, similitud {top1['similitud']})"
                )

            cd_pred = float(result["cd_pred"])
            cd_cont = float(result["cd_cont"])
            diferencia = cd_cont - cd_pred

            if diferencia > 0:
                st.caption("📈 Tendencia hacia un nivel superior.")
            elif diferencia < 0:
                st.caption("📉 Tendencia hacia un nivel inferior.")
            else:
                st.caption("➡️ El CD continuo coincide con el CD predicho.")

            if abs(diferencia) >= 0.7:
                st.warning("⚠️ El puesto está en una zona límite o próxima a otro nivel de CD.")

            st.caption(
                "El CD predicho se basa en el patrón más similar (Top-1), "
                "no en un redondeo del CD continuo."
            )

            st.subheader("TOP-K")
            st.json(result["top_k"])

            st.subheader("Explicación")
            st.json(result["explicacion"])

            st.subheader("Informe técnico")

            
            puesto_sugerido = obtener_puesto_orientativo_por_cd(result["cd_pred"])

            if "puesto_referencia_informe" not in st.session_state:
                st.session_state["puesto_referencia_informe"] = puesto_sugerido

            # Si cambia el CD, actualizamos la sugerencia solo si seguía con el valor anterior
            if st.session_state["puesto_referencia_informe"] in ("No indicado", "", puesto_sugerido):
                st.session_state["puesto_referencia_informe"] = puesto_sugerido

            puesto_referencia = st.text_input(
                "Puesto / referencia para el informe",
                key="puesto_referencia_informe"
            )

            try:
                html_informe = generar_html_informe(
                    result=result,
                    selected_rows=selected,
                    df_verbs=df_verbs,
                    configuracion_nombre=st.session_state.get("config_name", "No indicada"),
                    puesto_referencia=puesto_referencia,
                )

                st.markdown("### Vista previa del informe")
                st.components.v1.html(html_informe, height=600, scrolling=True)

                # Botón descarga
                st.download_button(
                    label="📄 Descargar informe HTML",
                    data=html_informe,
                    file_name=f"informe_tecnico_cd_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html",
                )
            except Exception as e:
                st.error(f"No se pudo generar el informe: {e}")
                if modo_tecnico:
                    st.exception(e)

    except Exception as e:
        st.error(f"Error en la clasificación: {e}")
        if modo_tecnico:
            st.exception(e)


# =========================================================
# TAB 2 — GESTIÓN DE VARIABLES
# =========================================================
with tab2:
    st.subheader("Gestión de variables")

    st.info(
        "Usa el botón de exportación para descargar la configuración completa en Excel (.xlsx). "
        "La exportación propia de las tablas genera CSV y no sirve como archivo completo de configuración."
    )

    st.markdown(
        "Aquí puedes modificar la configuración activa de la sesión. "
        "Después puedes aplicar cambios y exportar un nuevo Excel reutilizable."
    )

    try:
        excel_bytes = exportar_excel_configuracion(
            st.session_state.df_verbs,
            st.session_state.df_comp,
        )

        nombre_base = st.session_state.get("config_name", "configuracion")
        if nombre_base.lower().endswith(".xlsx"):
            nombre_base = nombre_base[:-5]

        st.download_button(
            label="📥 Exportar configuración completa en Excel (.xlsx)",
            data=excel_bytes,
            file_name=f"{nombre_base}_actualizada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"No se pudo preparar la exportación: {e}")

    st.markdown("### verbs_patterns")
    edited_verbs = st.data_editor(
        st.session_state.df_verbs,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_verbs",
    )

    st.markdown("### compatibility_verbs")
    edited_comp = st.data_editor(
        st.session_state.df_comp,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_comp",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("💾 Aplicar cambios", use_container_width=True):
            try:
                df_verbs_nuevo = cargar_hoja_verbs_desde_df(edited_verbs.copy())
                df_comp_nuevo = cargar_hoja_compat_desde_df(edited_comp.copy())

                faltantes = validar_valores_gestion(df_verbs_nuevo, df_comp_nuevo)

                st.session_state.df_verbs = df_verbs_nuevo
                st.session_state.df_comp = df_comp_nuevo
                st.session_state.verbos_faltantes = faltantes

                limpiar_checkboxes_clasificacion()

                st.success("Cambios aplicados correctamente a la sesión actual.")

            except Exception as e:
                st.error(f"No se pudieron aplicar los cambios: {e}")
                if modo_tecnico:
                    st.exception(e)

    with col_b:
        if st.button("↩️ Restaurar configuración cargada al inicio", use_container_width=True):
            st.session_state.df_verbs = st.session_state.df_verbs_base.copy()
            st.session_state.df_comp = st.session_state.df_comp_base.copy()

            limpiar_checkboxes_clasificacion()

            st.success("Se ha restaurado la configuración cargada al inicio de la sesión.")

    if modo_tecnico:
        with st.expander("Diagnóstico de gestión"):
            st.write("Columnas de patrón detectadas:", obtener_columnas_patron(st.session_state.df_verbs))
            st.write("Tamaño verbs_patterns:", st.session_state.df_verbs.shape)
            st.write("Tamaño compatibility_verbs:", st.session_state.df_comp.shape)