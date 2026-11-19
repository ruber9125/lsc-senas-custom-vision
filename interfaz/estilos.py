"""Paleta institucional y hoja de estilos (CSS) de la app."""

AZUL = "#003151"
AZUL_MEDIO = "#1D4E73"
DORADO = "#F7AB00"
TEXTO = "#1B2B3A"
TEXTO_SUAVE = "#5B6B7A"
FONDO_SUAVE = "#F3F6F9"
BORDE = "#DCE3EA"
VERDE = "#1E7B4F"
ROJO = "#B3261E"

# Colores de cada seña (distinguibles entre sí y con la paleta institucional)
COLORES = {"HOLA": "#2C6FA6", "BIEN": "#E39B00", "MAL": "#8E3B46"}

CSS = f"""
<style>
:root {{
  --azul: {AZUL}; --dorado: {DORADO}; --texto: {TEXTO}; --suave: {TEXTO_SUAVE};
  --fondo-suave: {FONDO_SUAVE}; --borde: {BORDE}; --verde: {VERDE}; --rojo: {ROJO};
}}
/* La barra superior de Streamlit Cloud mide 3.75rem y queda encima del contenido: se deja ese espacio libre */
.block-container {{ padding-top: 4.6rem; padding-bottom: 1rem; max-width: 1280px; }}

/* ---------- Encabezado ---------- */
.lsc-encabezado {{
  display: flex; align-items: center; gap: 1.6rem;
  padding: 1.1rem 1.4rem; margin-bottom: .6rem;
  background: #fff; border: 1px solid var(--borde); border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0, 49, 81, .06);
}}
.lsc-encabezado img {{ height: 64px; width: auto; flex-shrink: 0; }}
.lsc-encabezado .lsc-sep {{ width: 1px; align-self: stretch; background: var(--borde); }}
.lsc-encabezado h1 {{
  font-size: 1.75rem; line-height: 1.2; margin: 0; padding: 0; color: var(--azul); font-weight: 700;
}}
.lsc-encabezado .lsc-sub {{ color: var(--texto); font-size: 1rem; margin: .15rem 0 .45rem; }}
.lsc-encabezado .lsc-linea {{ width: 56px; height: 3px; background: var(--dorado); border-radius: 2px; margin-bottom: .45rem; }}
.lsc-encabezado .lsc-contexto {{ color: var(--suave); font-size: .9rem; margin: 0; }}

/* ---------- Pestañas ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom: 1px solid var(--borde); }}
.stTabs [data-baseweb="tab"] {{ padding: .55rem .9rem; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--dorado); height: 3px; }}

/* ---------- Títulos de sección ---------- */
.lsc-seccion {{
  display: flex; align-items: center; gap: .6rem; margin: 1.4rem 0 .6rem;
  color: var(--azul); font-weight: 700; font-size: 1.2rem;
}}
.lsc-seccion::before {{ content: ""; width: 4px; height: 1.2rem; background: var(--dorado); border-radius: 2px; }}
.lsc-nota {{ color: var(--suave); font-size: .88rem; margin: -.2rem 0 .8rem; }}
.lsc-etiqueta {{ font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--suave); margin-bottom: .35rem; }}

/* ---------- Indicador de pasos ---------- */
.lsc-pasos {{ display: flex; align-items: flex-start; margin: .8rem 0 1.2rem; }}
.lsc-paso {{ flex: 1; display: flex; flex-direction: column; align-items: center; position: relative; text-align: center; }}
.lsc-paso:not(:last-child)::after {{
  content: ""; position: absolute; top: 17px; left: calc(50% + 22px); right: calc(-50% + 22px);
  height: 2px; background: var(--borde);
}}
.lsc-paso.hecho:not(:last-child)::after {{ background: var(--azul); }}
.lsc-paso .lsc-circulo {{
  width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: .95rem; border: 2px solid var(--borde); background: #fff; color: var(--suave);
  z-index: 1;
}}
.lsc-paso.hecho .lsc-circulo {{ background: var(--azul); border-color: var(--azul); color: #fff; }}
.lsc-paso.actual .lsc-circulo {{ background: var(--dorado); border-color: var(--dorado); color: var(--azul);
  box-shadow: 0 0 0 5px rgba(247, 171, 0, .22); }}
.lsc-paso .lsc-nombre {{ margin-top: .45rem; font-size: .85rem; font-weight: 600; color: var(--suave); }}
.lsc-paso.hecho .lsc-nombre, .lsc-paso.actual .lsc-nombre {{ color: var(--azul); }}

/* ---------- Tarjetas ---------- */
.lsc-tarjeta {{
  background: #fff; border: 1px solid var(--borde); border-radius: 14px; padding: 1.1rem 1.2rem;
  box-shadow: 0 1px 6px rgba(0, 49, 81, .05);
}}
.lsc-tarjeta.suave {{ background: var(--fondo-suave); box-shadow: none; }}
.lsc-tarjeta h4 {{ margin: 0 0 .5rem; color: var(--azul); font-size: 1rem; }}
.lsc-tarjeta ul {{ margin: 0; padding-left: 1.1rem; }}
.lsc-tarjeta li {{ margin: .15rem 0; font-size: .93rem; }}
div[class*="st-key-panel_"] {{
  background: #fff; border: 1px solid var(--borde); border-radius: 14px; padding: 1rem 1.1rem;
  box-shadow: 0 1px 6px rgba(0, 49, 81, .05);
}}

/* ---------- Tarjeta de resultado ---------- */
.lsc-resultado {{
  background: #fff; border: 1px solid var(--borde); border-left: 6px solid var(--dorado);
  border-radius: 14px; padding: 1.1rem 1.3rem; box-shadow: 0 4px 16px rgba(0, 49, 81, .08);
}}
.lsc-chip {{
  display: inline-block; padding: .2rem .65rem; border-radius: 999px; font-size: .74rem; font-weight: 700;
  letter-spacing: .06em; text-transform: uppercase; background: rgba(247, 171, 0, .18); color: #7A5300;
}}
.lsc-sena {{ font-size: 3.6rem; font-weight: 800; line-height: 1.05; letter-spacing: .04em; margin: .35rem 0 .2rem; }}
.lsc-confianza {{ font-size: 1.05rem; color: var(--texto); }}
.lsc-confianza b {{ color: var(--azul); }}
.lsc-nivel {{ display: inline-block; margin-left: .4rem; padding: .05rem .5rem; border-radius: 999px; font-size: .75rem; font-weight: 700; }}
.lsc-nivel.alta {{ background: rgba(30, 123, 79, .12); color: var(--verde); }}
.lsc-nivel.media {{ background: rgba(247, 171, 0, .18); color: #7A5300; }}
.lsc-nivel.baja {{ background: rgba(179, 38, 30, .1); color: var(--rojo); }}
.lsc-desc {{ color: var(--suave); font-size: .92rem; margin: .5rem 0 .9rem; }}
.lsc-barra {{ display: grid; grid-template-columns: 3.4rem 1fr 3.6rem; align-items: center; gap: .6rem; margin: .4rem 0; }}
.lsc-barra .lsc-clase {{ font-weight: 700; font-size: .85rem; color: var(--texto); }}
.lsc-barra .lsc-pista {{ height: 12px; background: #EDF1F5; border-radius: 999px; overflow: hidden; }}
.lsc-barra .lsc-relleno {{ height: 100%; border-radius: 999px; }}
.lsc-barra .lsc-valor {{ text-align: right; font-variant-numeric: tabular-nums; font-size: .88rem; color: var(--texto); }}
.lsc-barra.atenuada .lsc-relleno {{ opacity: .45; }}
.lsc-barra.atenuada .lsc-clase, .lsc-barra.atenuada .lsc-valor {{ color: var(--suave); font-weight: 500; }}
.lsc-leyenda {{ display: flex; gap: 1rem; font-size: .82rem; color: var(--suave); margin-top: .4rem; }}
.lsc-punto {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .3rem; vertical-align: middle; }}

/* ---------- Confirmación ---------- */
div[class*="st-key-real_"] button {{
  min-height: 3.4rem; font-size: 1.1rem; font-weight: 700; border-width: 2px; background: #fff;
  transition: transform .08s ease, box-shadow .15s ease;
}}
div[class*="st-key-real_"] button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0, 49, 81, .12); }}
div[class*="st-key-real_HOLA"] button {{ border-color: {COLORES["HOLA"]}; color: {COLORES["HOLA"]}; }}
div[class*="st-key-real_BIEN"] button {{ border-color: {COLORES["BIEN"]}; color: #9A6A00; }}
div[class*="st-key-real_MAL"]  button {{ border-color: {COLORES["MAL"]};  color: {COLORES["MAL"]}; }}
div[class*="st-key-real_HOLA"] button:hover {{ background: {COLORES["HOLA"]}; color: #fff; }}
div[class*="st-key-real_BIEN"] button:hover {{ background: {COLORES["BIEN"]}; color: #fff; }}
div[class*="st-key-real_MAL"]  button:hover {{ background: {COLORES["MAL"]};  color: #fff; }}
.lsc-veredicto {{
  display: flex; align-items: center; gap: 1rem; padding: 1rem 1.2rem; border-radius: 14px; border: 1px solid;
}}
.lsc-veredicto .lsc-icono {{
  width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 1.4rem; font-weight: 800; flex-shrink: 0;
}}
.lsc-veredicto.acierto {{ background: rgba(30, 123, 79, .07); border-color: rgba(30, 123, 79, .35); }}
.lsc-veredicto.acierto .lsc-icono {{ background: var(--verde); }}
.lsc-veredicto.error {{ background: rgba(179, 38, 30, .06); border-color: rgba(179, 38, 30, .35); }}
.lsc-veredicto.error .lsc-icono {{ background: var(--rojo); }}
.lsc-veredicto .lsc-titulo {{ font-weight: 800; font-size: 1.1rem; }}
.lsc-veredicto.acierto .lsc-titulo {{ color: var(--verde); }}
.lsc-veredicto.error .lsc-titulo {{ color: var(--rojo); }}
.lsc-veredicto .lsc-detalle {{ color: var(--texto); font-size: .95rem; }}

/* ---------- KPI ---------- */
.lsc-kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .8rem; margin: .4rem 0 1rem; }}
.lsc-kpi {{
  background: #fff; border: 1px solid var(--borde); border-top: 4px solid var(--azul); border-radius: 12px;
  padding: .85rem 1rem; box-shadow: 0 1px 6px rgba(0, 49, 81, .05);
}}
.lsc-kpi.destacado {{ border-top-color: var(--dorado); }}
.lsc-kpi .lsc-kpi-nombre {{ font-size: .78rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--suave); }}
.lsc-kpi .lsc-kpi-valor {{ font-size: 1.9rem; font-weight: 800; color: var(--azul); line-height: 1.2; font-variant-numeric: tabular-nums; }}
.lsc-kpi .lsc-kpi-ayuda {{ font-size: .76rem; color: var(--suave); }}

/* ---------- Chips de información ---------- */
.lsc-chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .6rem; }}
.lsc-chips span {{
  background: var(--fondo-suave); border: 1px solid var(--borde); border-radius: 999px;
  padding: .2rem .7rem; font-size: .83rem; color: var(--texto);
}}
.lsc-chips b {{ color: var(--azul); }}

/* ---------- Pestaña "Cómo funciona" ---------- */
.lsc-flujo {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .8rem; margin: .4rem 0 1rem; }}
.lsc-etapa {{ background: #fff; border: 1px solid var(--borde); border-radius: 12px; padding: 1rem; position: relative; }}
.lsc-etapa .lsc-num {{
  width: 30px; height: 30px; border-radius: 50%; background: var(--azul); color: #fff; font-weight: 700;
  display: flex; align-items: center; justify-content: center; margin-bottom: .5rem; font-size: .9rem;
}}
.lsc-etapa h5 {{ margin: 0 0 .3rem; color: var(--azul); font-size: .98rem; padding: 0; }}
.lsc-etapa p {{ margin: 0; font-size: .86rem; color: var(--suave); }}
.lsc-cifras {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .4rem 0 1rem; }}
.lsc-cifra {{ background: var(--fondo-suave); border-radius: 12px; padding: .9rem 1rem; text-align: center; }}
.lsc-cifra b {{ display: block; font-size: 2rem; color: var(--azul); line-height: 1.1; }}
.lsc-cifra span {{ font-size: .85rem; color: var(--suave); }}
.lsc-aviso {{ border-left: 5px solid var(--dorado); background: #FFFBF0; border-radius: 10px; padding: .9rem 1.1rem; }}
.lsc-aviso ul {{ margin: .2rem 0 0; padding-left: 1.1rem; }}
.lsc-vacio {{
  text-align: center; padding: 2.2rem 1rem; border: 2px dashed var(--borde); border-radius: 14px;
  color: var(--suave); background: var(--fondo-suave);
}}
.lsc-vacio b {{ display: block; color: var(--azul); font-size: 1.1rem; margin-bottom: .3rem; }}

/* ---------- Barra lateral ---------- */
.lsc-estado {{
  display: flex; align-items: center; gap: .55rem; padding: .6rem .8rem; border-radius: 10px;
  font-size: .9rem; font-weight: 600; border: 1px solid;
}}
.lsc-estado.ok {{ background: rgba(30, 123, 79, .08); border-color: rgba(30, 123, 79, .3); color: var(--verde); }}
.lsc-estado.mal {{ background: rgba(179, 38, 30, .07); border-color: rgba(179, 38, 30, .3); color: var(--rojo); }}
.lsc-estado .lsc-led {{ width: 9px; height: 9px; border-radius: 50%; background: currentColor; box-shadow: 0 0 0 3px rgba(0,0,0,.05); }}
.lsc-lateral-titulo {{ font-size: .75rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--suave); margin: 1rem 0 .4rem; }}
.lsc-ficha {{ font-size: .9rem; line-height: 1.6; }}
.lsc-ficha b {{ color: var(--azul); }}
.lsc-contadores {{ display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-bottom: .6rem; }}
.lsc-contador {{ background: #fff; border: 1px solid var(--borde); border-radius: 10px; padding: .55rem .7rem; }}
.lsc-contador b {{ display: block; font-size: 1.5rem; color: var(--azul); line-height: 1.1; }}
.lsc-contador span {{ font-size: .75rem; color: var(--suave); }}

/* Barra lateral: sin el espacio vacío superior, para que la tarjeta del proyecto quede a la misma
   altura que el encabezado de la página. */
[data-testid="stSidebarHeader"] {{ height: 2.2rem; min-height: 0; padding-top: .4rem; padding-bottom: 0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0; }}
.lsc-proyecto {{
  background: #fff; border: 1px solid var(--borde); border-radius: 14px; padding: 1rem .9rem;
  text-align: center; box-shadow: 0 2px 10px rgba(0, 49, 81, .06);
}}
.lsc-proyecto-etiqueta {{ font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--suave); }}
.lsc-proyecto-nombre {{ font-size: 1.2rem; font-weight: 800; color: var(--azul); line-height: 1.2; margin-top: .15rem; }}
.lsc-proyecto-sep {{ width: 40px; height: 3px; background: var(--dorado); border-radius: 2px; margin: .6rem auto .55rem; }}
.lsc-proyecto-autor {{ font-size: .86rem; font-weight: 700; color: var(--texto); line-height: 1.3; }}
.lsc-proyecto-inst {{ font-size: .78rem; color: var(--suave); margin-top: .1rem; }}

/* ---------- Banner "Cómo funciona" ---------- */
div[class*="st-key-banner_flujo"] {{
  background: #fff; border: 1px solid var(--borde); border-radius: 14px; padding: .6rem .6rem .2rem;
  box-shadow: 0 1px 6px rgba(0, 49, 81, .05); margin-top: .4rem;
}}
div[class*="st-key-banner_flujo"] img {{ border-radius: 10px; }}
.lsc-banner-pie {{ text-align: center; font-size: .85rem; color: var(--suave); font-style: italic; margin: .2rem 0 .3rem; }}

.lsc-privacidad {{
  background: #fff; border: 1px solid var(--borde); border-left: 4px solid var(--azul);
  border-radius: 10px; padding: .7rem .85rem;
}}
.lsc-privacidad-titulo {{ display: flex; align-items: center; gap: .45rem; font-weight: 700; color: var(--azul);
  font-size: .9rem; margin-bottom: .35rem; }}
.lsc-candado {{ width: 18px; height: 18px; border-radius: 50%; background: var(--azul); color: #fff;
  font-size: .7rem; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
.lsc-privacidad p {{ margin: .2rem 0 0; font-size: .85rem; line-height: 1.45; color: var(--texto); }}
.lsc-privacidad b {{ color: var(--azul); }}

/* ---------- Pie de página ---------- */
.lsc-pie {{
  margin-top: 2.5rem; padding: 1.1rem 0 .4rem; border-top: 1px solid var(--borde);
  text-align: center; color: var(--suave); font-size: .85rem;
}}
.lsc-pie b {{ color: var(--azul); }}
.lsc-pie .lsc-linea {{ width: 40px; height: 3px; background: var(--dorado); border-radius: 2px; margin: 0 auto .6rem; }}

/* ---------- Pantallas medianas ---------- */
@media (max-width: 1000px) {{
  .lsc-kpis {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  .lsc-flujo {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
/* ---------- Pantallas pequeñas ---------- */
@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 4.4rem; }}
  .lsc-encabezado {{ flex-direction: column; text-align: center; gap: .7rem; padding: 1rem; }}
  .lsc-encabezado .lsc-sep {{ display: none; }}
  .lsc-encabezado img {{ height: 48px; }}
  .lsc-encabezado h1 {{ font-size: 1.35rem; }}
  .lsc-encabezado .lsc-linea {{ margin-left: auto; margin-right: auto; }}
  .lsc-pasos {{ flex-direction: column; gap: .5rem; }}
  .lsc-paso {{ flex-direction: row; gap: .7rem; text-align: left; }}
  .lsc-paso:not(:last-child)::after {{ display: none; }}
  .lsc-paso .lsc-nombre {{ margin-top: 0; }}
  .lsc-sena {{ font-size: 2.8rem; }}
  .lsc-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .lsc-kpi .lsc-kpi-valor {{ font-size: 1.5rem; }}
  .lsc-flujo, .lsc-cifras {{ grid-template-columns: 1fr 1fr; }}
  .lsc-veredicto {{ flex-direction: column; text-align: center; }}
}}
</style>
"""
