"""
Sistema de Clasificación de Puestos por Similitud Coseno Ponderada
==================================================================
Autor  : Senior Python Developer / Applied Data Scientist
Versión: 1.0.0
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# 49 verbos que forman el espacio dimensional
VERBOS_VALIDOS: List[str] = [
    "administrar",
    "anticipar",
    "representar",
    "definir",
    "aprobar",
    "coordinar",
    "decidir",
    "establecer",
    "evaluar",
    "fomentar",
    "comunicar",
    "planificar",
    "dirigir",
    "motivar",
    "sustentar",
    "colaborar",
    "controlar",
    "retroalimentar",
    "desarrollar",
    "programar",
    "verificar",
    "orientar",
    "proponer",
    "analizar",
    "asesorar",
    "documentar",
    "elaborar",
    "estudiar",
    "identificar",
    "investigar",
    "informar",
    "sistematizar",
    "organizar",
    "supervisar",
    "implementar",
    "facilitar",
    "promover",
    "apoyar",
    "asistir",
    "revisar",
    "optimizar",
    "ejecutar",
    "reportar",
    "participar",
    "aportar",
    "aprender",
    "atender",
    "capacitar",
    "cumplir",
]

assert len(VERBOS_VALIDOS) == 49, "El espacio dimensional debe tener exactamente 49 verbos."

VERBO_INDEX: Dict[str, int] = {v: i for i, v in enumerate(VERBOS_VALIDOS)}

# Categorías jerárquicas y sus pesos
CATEGORIA_PESO: Dict[str, float] = {
    "nuclear":    2.50,
    "relevante":  1.65,
    "apoyo":      1.15,
    "accesorio":  0.80,
}

CATEGORIAS_VALIDAS = set(CATEGORIA_PESO.keys())

# Parámetros globales configurables
ALPHA_DEFAULT: float = 0.7
TOPK_DEFAULT:  int   = 3
EPS:           float = 1e-6


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Verbo:
    """Representa un verbo con su categoría jerárquica."""
    nombre:    str
    categoria: str

    def __post_init__(self) -> None:
        if self.nombre not in VERBO_INDEX:
            raise ValueError(
                f"Verbo inválido: '{self.nombre}'. "
                f"Debe ser uno de los 49 verbos definidos."
            )
        if self.categoria not in CATEGORIAS_VALIDAS:
            raise ValueError(
                f"Categoría inválida: '{self.categoria}'. "
                f"Opciones: {sorted(CATEGORIAS_VALIDAS)}"
            )

    @property
    def peso_jerarquico(self) -> float:
        return CATEGORIA_PESO[self.categoria]

    @property
    def indice(self) -> int:
        return VERBO_INDEX[self.nombre]


@dataclass
class Patron:
    """
    Patrón de puesto de trabajo.

    Attributes
    ----------
    id      : Identificador único.
    nombre  : Nombre del puesto.
    cd      : Código de Denominación (entero o float).
    verbos  : Lista de Verbo que definen el patrón.
    """
    id:     str
    nombre: str
    cd:     float
    verbos: List[Verbo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.verbos:
            raise ValueError(f"El patrón '{self.id}' debe tener al menos un verbo.")
        nombres = [v.nombre for v in self.verbos]
        if len(nombres) != len(set(nombres)):
            raise ValueError(
                f"El patrón '{self.id}' contiene verbos duplicados."
            )

    def vector_binario(self) -> List[int]:
        """Vector binario de 49 posiciones."""
        vec = [0] * 49
        for v in self.verbos:
            vec[v.indice] = 1
        return vec


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

def compute_df(patrones: List[Patron]) -> List[int]:
    """
    Calcula la frecuencia de documento (df) para cada verbo.

    df_i = número de patrones que contienen el verbo i.
    Garantiza df_i >= 1 (mínimo 1 para evitar log(0)).

    Returns
    -------
    Lista de 49 enteros con df por verbo.
    """
    if not patrones:
        raise ValueError("La lista de patrones no puede estar vacía.")

    df = [0] * 49
    for patron in patrones:
        vec = patron.vector_binario()
        for i, val in enumerate(vec):
            if val == 1:
                df[i] += 1

    # Garantizar df_i >= 1
    df = [max(d, 1) for d in df]
    return df


def compute_weights(
    patrones:  List[Patron],
    alpha:     float = ALPHA_DEFAULT,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calcula los tres tipos de pesos.

    Pesos jerárquicos se derivan de la categoría asignada a cada verbo
    en CADA patrón. Si un verbo aparece con distintas categorías en
    distintos patrones, se usa el máximo peso (política conservadora).
    Para el usuario se aplica peso de categoría máxima observada.

    Returns
    -------
    w_comp : List[float] — peso jerárquico por verbo (49 valores)
    w_idf  : List[float] — peso IDF por verbo (49 valores)
    w      : List[float] — peso total = w_comp * w_idf (49 valores)
    """
    N = len(patrones)
    df = compute_df(patrones)

    # Peso jerárquico: máximo observado en todos los patrones
    w_comp = [0.80] * 49  # accesorio por defecto si no aparece en ningún patrón
    for patron in patrones:
        for verbo in patron.verbos:
            i = verbo.indice
            w_comp[i] = max(w_comp[i], verbo.peso_jerarquico)

    # Peso IDF
    w_idf = [(math.log(N / df[i]) + 1) ** alpha for i in range(49)]

    # Peso total
    w = [w_comp[i] * w_idf[i] for i in range(49)]

    return w_comp, w_idf, w


def cosine(u: List[float], p: List[float]) -> float:
    """
    Similitud coseno entre dos vectores ponderados.

    Raises
    ------
    ZeroDivisionError si la norma del vector usuario es 0.
    ValueError si los vectores tienen distinto tamaño.
    """
    if len(u) != len(p):
        raise ValueError(
            f"Vectores de distinto tamaño: {len(u)} vs {len(p)}"
        )

    dot    = sum(ui * pi for ui, pi in zip(u, p))
    norm_u = math.sqrt(sum(ui ** 2 for ui in u))
    norm_p = math.sqrt(sum(pi ** 2 for pi in p))

    if norm_u == 0.0:
        raise ZeroDivisionError(
            "El vector del usuario tiene norma 0. "
            "El usuario no ha seleccionado ningún verbo válido."
        )
    if norm_p == 0.0:
        # Patrón vacío en el espacio ponderado (no debería ocurrir con validaciones)
        return 0.0

    return dot / (norm_u * norm_p)


def classify(
    verbos_usuario:   List[Verbo],
    patrones:         List[Patron],
    alpha:            float = ALPHA_DEFAULT,
    topk:             int   = TOPK_DEFAULT,
) -> Dict:
    """
    Clasifica al usuario entre los patrones disponibles.

    Parameters
    ----------
    verbos_usuario : Verbos seleccionados por el usuario (con categorías).
    patrones       : Lista de patrones de referencia.
    alpha          : Exponente IDF.
    topk           : Número máximo de mejores coincidencias a devolver.

    Returns
    -------
    Diccionario con:
        cd_pred      : float  — CD discreto predicho.
        cd_cont      : float  — CD continuo (promedio ponderado Top-K).
        top_k        : list   — [(patron_id, nombre, cd, similitud), ...]
        explicacion  : dict   — verbos comunes, solo_usuario, solo_top1.
    """
    # ── Validaciones ───────────────────────────────────────────────────────
    if not patrones:
        raise ValueError("Se requiere al menos un patrón de referencia.")

    topk_efectivo = min(topk, len(patrones))

    # Validar verbos de usuario
    nombres_usuario = [v.nombre for v in verbos_usuario]
    if len(nombres_usuario) != len(set(nombres_usuario)):
        raise ValueError("El usuario tiene verbos duplicados.")

    # ── Pesos ──────────────────────────────────────────────────────────────
    _, _, w = compute_weights(patrones, alpha=alpha)

    # ── Vector usuario ─────────────────────────────────────────────────────
    s = [0] * 49
    for verbo in verbos_usuario:
        s[verbo.indice] = 1

    u = [w[i] * s[i] for i in range(49)]

    # ── Vectores de patrones ponderados ────────────────────────────────────
    patrones_vecs: List[Tuple[Patron, List[float]]] = []
    for patron in patrones:
        t = patron.vector_binario()
        p_vec = [w[i] * t[i] for i in range(49)]
        patrones_vecs.append((patron, p_vec))

    # ── Similitudes ────────────────────────────────────────────────────────
    similitudes: List[Tuple[float, Patron]] = []
    for patron, p_vec in patrones_vecs:
        sim = cosine(u, p_vec)
        similitudes.append((sim, patron))

    similitudes.sort(key=lambda x: x[0], reverse=True)
    top_k_raw = similitudes[:topk_efectivo]

    # ── CD continuo ────────────────────────────────────────────────────────
    suma_sim    = sum(sim for sim, _ in top_k_raw)
    if suma_sim == 0.0:
        # Todos los patrones tienen similitud 0; usar top-1 CD
        cd_cont = top_k_raw[0][1].cd
    else:
        cd_cont = sum(sim * patron.cd for sim, patron in top_k_raw) / suma_sim

    # ── CD discreto ────────────────────────────────────────────────────────
    cds_existentes = {patron.cd for patron in patrones}

    cd_pred: Optional[float] = None
    for cd in cds_existentes:
        if abs(cd_cont - cd) < EPS:
            cd_pred = cd
            break

    if cd_pred is None:
        # Top-1 del Top-K
        cd_pred = top_k_raw[0][1].cd

    # ── Top-K formateado ───────────────────────────────────────────────────
    top_k_out = [
        {
            "patron_id":  patron.id,
            "nombre":     patron.nombre,
            "cd":         patron.cd,
            "similitud":  round(sim, 6),
        }
        for sim, patron in top_k_raw
    ]

    # ── Explicación ────────────────────────────────────────────────────────
    top1_patron = top_k_raw[0][1]
    set_usuario = {v.nombre for v in verbos_usuario}
    set_top1    = {v.nombre for v in top1_patron.verbos}

    comunes         = sorted(set_usuario & set_top1)
    solo_usuario    = sorted(set_usuario - set_top1)
    solo_top1       = sorted(set_top1    - set_usuario)

    explicacion = {
        "top1_patron_id":   top1_patron.id,
        "top1_nombre":      top1_patron.nombre,
        "verbos_comunes":   comunes,
        "solo_en_usuario":  solo_usuario,
        "solo_en_top1":     solo_top1,
    }

    return {
        "cd_pred":     cd_pred,
        "cd_cont":     round(cd_cont, 6),
        "top_k":       top_k_out,
        "explicacion": explicacion,
    }


# ---------------------------------------------------------------------------
# Dataset de ejemplo
# ---------------------------------------------------------------------------

def build_dataset() -> List[Patron]:
    """
    Construye un dataset pequeño de 7 patrones de puestos.
    CD representa el nivel jerárquico (1=operativo … 5=directivo).
    """
    patrones = [
        Patron(
            id="P01", nombre="Director General", cd=5.0,
            verbos=[
                Verbo("planificar",   "nuclear"),
                Verbo("dirigir",      "nuclear"),
                Verbo("liderar",      "nuclear"),
                Verbo("delegar",      "relevante"),
                Verbo("aprobar",      "relevante"),
                Verbo("negociar",     "relevante"),
                Verbo("presupuestar", "apoyo"),
                Verbo("reportar",     "accesorio"),
            ],
        ),
        Patron(
            id="P02", nombre="Gerente de Área", cd=4.0,
            verbos=[
                Verbo("coordinar",   "nuclear"),
                Verbo("supervisar",  "nuclear"),
                Verbo("gestionar",   "nuclear"),
                Verbo("evaluar",     "relevante"),
                Verbo("planificar",  "relevante"),
                Verbo("comunicar",   "apoyo"),
                Verbo("reportar",    "apoyo"),
                Verbo("delegar",     "accesorio"),
            ],
        ),
        Patron(
            id="P03", nombre="Jefe de Departamento", cd=3.0,
            verbos=[
                Verbo("organizar",   "nuclear"),
                Verbo("controlar",   "nuclear"),
                Verbo("supervisar",  "relevante"),
                Verbo("capacitar",   "relevante"),
                Verbo("evaluar",     "relevante"),
                Verbo("documentar",  "apoyo"),
                Verbo("validar",     "apoyo"),
                Verbo("registrar",   "accesorio"),
            ],
        ),
        Patron(
            id="P04", nombre="Analista Senior", cd=2.0,
            verbos=[
                Verbo("analizar",     "nuclear"),
                Verbo("investigar",   "nuclear"),
                Verbo("diseñar",      "nuclear"),
                Verbo("desarrollar",  "relevante"),
                Verbo("elaborar",     "relevante"),
                Verbo("documentar",   "apoyo"),
                Verbo("proponer",     "apoyo"),
                Verbo("verificar",    "accesorio"),
            ],
        ),
        Patron(
            id="P05", nombre="Técnico Especialista", cd=2.0,
            verbos=[
                Verbo("implementar",  "nuclear"),
                Verbo("ejecutar",     "nuclear"),
                Verbo("operar",       "nuclear"),
                Verbo("mantener",     "relevante"),
                Verbo("diagnosticar", "relevante"),
                Verbo("instalar",     "apoyo"),
                Verbo("verificar",    "apoyo"),
                Verbo("registrar",    "accesorio"),
            ],
        ),
        Patron(
            id="P06", nombre="Asistente Administrativo", cd=1.0,
            verbos=[
                Verbo("procesar",    "nuclear"),
                Verbo("registrar",   "nuclear"),
                Verbo("archivar",    "nuclear"),
                Verbo("clasificar",  "relevante"),
                Verbo("atender",     "relevante"),
                Verbo("documentar",  "apoyo"),
                Verbo("comunicar",   "accesorio"),
            ],
        ),
        Patron(
            id="P07", nombre="Auditor Interno", cd=3.0,
            verbos=[
                Verbo("auditar",    "nuclear"),
                Verbo("revisar",    "nuclear"),
                Verbo("verificar",  "nuclear"),
                Verbo("analizar",   "relevante"),
                Verbo("evaluar",    "relevante"),
                Verbo("reportar",   "apoyo"),
                Verbo("certificar", "apoyo"),
                Verbo("documentar", "accesorio"),
            ],
        ),
    ]
    return patrones


# ---------------------------------------------------------------------------
# Tests unitarios
# ---------------------------------------------------------------------------

class TestComputeDf(unittest.TestCase):

    def setUp(self):
        self.patrones = build_dataset()

    def test_df_length(self):
        df = compute_df(self.patrones)
        self.assertEqual(len(df), 49)

    def test_df_minimum_one(self):
        df = compute_df(self.patrones)
        for val in df:
            self.assertGreaterEqual(val, 1)

    def test_df_planificar(self):
        # "planificar" aparece en P01 y P02
        df = compute_df(self.patrones)
        idx = VERBO_INDEX["planificar"]
        self.assertEqual(df[idx], 2)

    def test_empty_patrones_raises(self):
        with self.assertRaises(ValueError):
            compute_df([])


class TestComputeWeights(unittest.TestCase):

    def setUp(self):
        self.patrones = build_dataset()

    def test_weights_length(self):
        w_comp, w_idf, w = compute_weights(self.patrones)
        self.assertEqual(len(w_comp), 49)
        self.assertEqual(len(w_idf),  49)
        self.assertEqual(len(w),      49)

    def test_weights_positive(self):
        _, _, w = compute_weights(self.patrones)
        for val in w:
            self.assertGreater(val, 0)

    def test_alpha_effect(self):
        _, w_idf_07, _ = compute_weights(self.patrones, alpha=0.7)
        _, w_idf_10, _ = compute_weights(self.patrones, alpha=1.0)
        # Con alpha mayor, el IDF de términos raros debe ser mayor
        idx = VERBO_INDEX["liderar"]  # solo en P01
        self.assertGreater(w_idf_10[idx], w_idf_07[idx])


class TestCosine(unittest.TestCase):

    def test_identical_vectors(self):
        u = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine(u, u), 1.0, places=10)

    def test_orthogonal_vectors(self):
        u = [1.0, 0.0]
        p = [0.0, 1.0]
        self.assertAlmostEqual(cosine(u, p), 0.0, places=10)

    def test_zero_user_vector_raises(self):
        with self.assertRaises(ZeroDivisionError):
            cosine([0.0, 0.0], [1.0, 2.0])

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine([1.0, 2.0], [1.0])

    def test_range(self):
        import random
        random.seed(42)
        u = [random.random() for _ in range(49)]
        p = [random.random() for _ in range(49)]
        sim = cosine(u, p)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim,  1.0)


class TestVerboDataclass(unittest.TestCase):

    def test_valid_verbo(self):
        v = Verbo("planificar", "nuclear")
        self.assertEqual(v.peso_jerarquico, 2.50)
        self.assertEqual(v.indice, VERBO_INDEX["planificar"])

    def test_invalid_nombre_raises(self):
        with self.assertRaises(ValueError):
            Verbo("volar", "nuclear")

    def test_invalid_categoria_raises(self):
        with self.assertRaises(ValueError):
            Verbo("planificar", "superimportante")


class TestPatronDataclass(unittest.TestCase):

    def test_valid_patron(self):
        p = Patron(
            id="T01", nombre="Test", cd=1.0,
            verbos=[Verbo("planificar", "nuclear")],
        )
        self.assertEqual(len(p.vector_binario()), 49)

    def test_empty_verbos_raises(self):
        with self.assertRaises(ValueError):
            Patron(id="T01", nombre="Test", cd=1.0, verbos=[])

    def test_duplicate_verbos_raises(self):
        with self.assertRaises(ValueError):
            Patron(
                id="T01", nombre="Test", cd=1.0,
                verbos=[
                    Verbo("planificar", "nuclear"),
                    Verbo("planificar", "relevante"),
                ],
            )

    def test_vector_binario_correct(self):
        v = Verbo("planificar", "nuclear")
        p = Patron(id="T01", nombre="Test", cd=1.0, verbos=[v])
        vec = p.vector_binario()
        self.assertEqual(vec[VERBO_INDEX["planificar"]], 1)
        self.assertEqual(sum(vec), 1)


class TestClassify(unittest.TestCase):

    def setUp(self):
        self.patrones = build_dataset()

    def test_output_keys(self):
        usuario = [
            Verbo("planificar",  "nuclear"),
            Verbo("dirigir",     "nuclear"),
            Verbo("liderar",     "nuclear"),
            Verbo("negociar",    "relevante"),
        ]
        result = classify(usuario, self.patrones)
        self.assertIn("cd_pred",     result)
        self.assertIn("cd_cont",     result)
        self.assertIn("top_k",       result)
        self.assertIn("explicacion", result)

    def test_cd_pred_in_existing_cds(self):
        usuario = [
            Verbo("analizar",   "nuclear"),
            Verbo("investigar", "nuclear"),
            Verbo("documentar", "apoyo"),
        ]
        result   = classify(usuario, self.patrones)
        cds_existentes = {p.cd for p in self.patrones}
        self.assertIn(result["cd_pred"], cds_existentes)

    def test_topk_length(self):
        usuario = [Verbo("planificar", "nuclear")]
        result  = classify(usuario, self.patrones, topk=3)
        self.assertLessEqual(len(result["top_k"]), 3)

    def test_top_k_sorted_descending(self):
        usuario = [
            Verbo("organizar",  "nuclear"),
            Verbo("controlar",  "nuclear"),
            Verbo("supervisar", "relevante"),
        ]
        result = classify(usuario, self.patrones)
        sims   = [item["similitud"] for item in result["top_k"]]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_explicacion_keys(self):
        usuario = [Verbo("planificar", "nuclear")]
        result  = classify(usuario, self.patrones)
        exp     = result["explicacion"]
        self.assertIn("verbos_comunes",  exp)
        self.assertIn("solo_en_usuario", exp)
        self.assertIn("solo_en_top1",    exp)

    def test_zero_user_vector_raises(self):
        # Usuario con verbos que NO están en el sistema — imposible por validación de Verbo,
        # pero si s es todo ceros en la ponderación no puede ocurrir con verbos válidos.
        # Simulamos directamente:
        with self.assertRaises((ZeroDivisionError, ValueError)):
            cosine([0.0] * 49, [1.0] * 49)

    def test_empty_patrones_raises(self):
        usuario = [Verbo("planificar", "nuclear")]
        with self.assertRaises(ValueError):
            classify(usuario, [])

    def test_topk_larger_than_n(self):
        # Solo 2 patrones, topk=3 → debe usar 2
        patrones_mini = build_dataset()[:2]
        usuario = [Verbo("planificar", "nuclear")]
        result  = classify(usuario, patrones_mini, topk=3)
        self.assertLessEqual(len(result["top_k"]), 2)


# ---------------------------------------------------------------------------
# Función principal / Demo
# ---------------------------------------------------------------------------

def _print_separator(char: str = "─", width: int = 65) -> None:
    print(char * width)


def _print_result(result: Dict, label: str = "") -> None:
    _print_separator()
    if label:
        print(f"  CASO: {label}")
        _print_separator()
    print(f"  CD Continuo  : {result['cd_cont']:.4f}")
    print(f"  CD Predicho  : {result['cd_pred']}")
    print()
    print("  TOP-K resultados:")
    for i, item in enumerate(result["top_k"], 1):
        print(
            f"    {i}. [{item['patron_id']}] {item['nombre']:<30} "
            f"CD={item['cd']}  sim={item['similitud']:.4f}"
        )
    print()
    exp = result["explicacion"]
    print(f"  Top-1 patrón : [{exp['top1_patron_id']}] {exp['top1_nombre']}")
    print(f"  Verbos comunes    ({len(exp['verbos_comunes']):2d}): "
          f"{', '.join(exp['verbos_comunes']) or '—'}")
    print(f"  Solo en usuario   ({len(exp['solo_en_usuario']):2d}): "
          f"{', '.join(exp['solo_en_usuario']) or '—'}")
    print(f"  Solo en Top-1     ({len(exp['solo_en_top1']):2d}): "
          f"{', '.join(exp['solo_en_top1']) or '—'}")
    _print_separator()


def main() -> None:
    print()
    print("=" * 65)
    print("  SISTEMA DE CLASIFICACIÓN DE PUESTOS")
    print("  Similitud Coseno Ponderada (IDF + Jerarquía)")
    print("=" * 65)

    patrones = build_dataset()

    print(f"\n  Dataset: {len(patrones)} patrones cargados.")
    print(f"  Dimensiones: {len(VERBOS_VALIDOS)} verbos")
    print(f"  Alpha IDF: {ALPHA_DEFAULT}  |  TopK: {TOPK_DEFAULT}")

    # ── Caso 1: Perfil directivo ───────────────────────────────────────────
    usuario_1 = [
        Verbo("planificar",   "nuclear"),
        Verbo("dirigir",      "nuclear"),
        Verbo("liderar",      "nuclear"),
        Verbo("delegar",      "relevante"),
        Verbo("negociar",     "relevante"),
        Verbo("presupuestar", "apoyo"),
    ]
    res1 = classify(usuario_1, patrones)
    _print_result(res1, label="Perfil Directivo")

    # ── Caso 2: Perfil analítico / técnico ────────────────────────────────
    usuario_2 = [
        Verbo("analizar",    "nuclear"),
        Verbo("investigar",  "nuclear"),
        Verbo("diseñar",     "relevante"),
        Verbo("desarrollar", "relevante"),
        Verbo("documentar",  "apoyo"),
        Verbo("verificar",   "accesorio"),
    ]
    res2 = classify(usuario_2, patrones)
    _print_result(res2, label="Perfil Analítico")

    # ── Caso 3: Perfil operativo ───────────────────────────────────────────
    usuario_3 = [
        Verbo("procesar",   "nuclear"),
        Verbo("registrar",  "nuclear"),
        Verbo("archivar",   "nuclear"),
        Verbo("clasificar", "relevante"),
        Verbo("atender",    "relevante"),
    ]
    res3 = classify(usuario_3, patrones)
    _print_result(res3, label="Perfil Operativo")

    # ── Caso 4: Perfil mixto / auditoria ──────────────────────────────────
    usuario_4 = [
        Verbo("auditar",    "nuclear"),
        Verbo("revisar",    "nuclear"),
        Verbo("analizar",   "relevante"),
        Verbo("evaluar",    "relevante"),
        Verbo("supervisar", "apoyo"),
        Verbo("reportar",   "accesorio"),
    ]
    res4 = classify(usuario_4, patrones)
    _print_result(res4, label="Perfil Auditoría / Control")

    # ── Pesos IDF (diagnóstico) ────────────────────────────────────────────
    print("\n  Diagnóstico IDF — verbos con mayor discriminación:\n")
    _, w_idf, w = compute_weights(patrones)
    ranked = sorted(
        [(VERBOS_VALIDOS[i], w_idf[i], w[i]) for i in range(49)],
        key=lambda x: x[1], reverse=True
    )
    print(f"  {'Verbo':<20} {'w_idf':>8}  {'w_total':>8}")
    _print_separator("-", 42)
    for nombre, widf, wtotal in ranked[:10]:
        print(f"  {nombre:<20} {widf:>8.4f}  {wtotal:>8.4f}")
    print()

    # ── Tests unitarios ────────────────────────────────────────────────────
    print("  Ejecutando tests unitarios...\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestComputeDf, TestComputeWeights, TestCosine,
        TestVerboDataclass, TestPatronDataclass, TestClassify,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
