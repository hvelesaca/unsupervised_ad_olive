import numpy as np
import rasterio
import rasterio.transform
import glob
import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from typing import Optional, List, Dict, Tuple
from pyproj import Transformer

from PIL import Image, ImageDraw, ImageFont
import pickle

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ----
# GEOJSON DE PARCELAS (embebido)
# ----

GEOJSON_STR = """{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0454172302099902,38.133864190026486],[-3.04538389621728,38.13389102469108],[-3.0452670281725833,38.13398510401609],[-3.04537473982571,38.13415621471216],[-3.045572701950234,38.13449393586032],[-3.045718525281162,38.13484212935893],[-3.0457698384572467,38.1349646828489],[-3.045819890995019,38.135079395067976],[-3.0459113166093124,38.13529530368643],[-3.0461212631797054,38.13580606085395],[-3.044932121667691,38.13616447692229],[-3.0442996818899912,38.13480731176929],[-3.0442649615304145,38.13475586196816],[-3.0443479105255253,38.13474375598469],[-3.0444294798592915,38.13471560787576],[-3.0444850300876807,38.134681520349915],[-3.044530636863746,38.134623372569564],[-3.044599728057517,38.13452970647242],[-3.044705563002411,38.13425514464484],[-3.0449440021779535,38.13363733380362],[-3.0454172302099902,38.133864190026486]]]},"properties":{"dn_surface":24060.584774877367,"provincia":23,"municipio":48,"agregado":0,"zona":0,"poligono":20,"parcela":112,"recinto":1,"uso_sigpac":"OV","idPanel":"23:48:0:0:20:112:1"},"id":1326710838},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.016743861764303,38.13111071874488],[-3.0173083670509375,38.13121071234113],[-3.0173418611380076,38.13132663574527],[-3.017493817507852,38.13140268978643],[-3.0176060922693364,38.13142169407661],[-3.018397343975892,38.13148920863432],[-3.018520725597931,38.131535624264075],[-3.0185272006185175,38.13156911918935],[-3.0185361198022136,38.131568670757474],[-3.01854344055688,38.13165313100984],[-3.0185620651468374,38.131867997757666],[-3.0195019112416746,38.132087417555596],[-3.0203982393308424,38.1323023312421],[-3.0202160343470297,38.132535531727854],[-3.020156693821799,38.13260216031833],[-3.0198114339894917,38.132989828392255],[-3.019764094673937,38.133040756838966],[-3.0197390428402446,38.133032828396296],[-3.0181555319223126,38.13253167605217],[-3.0177521318533795,38.132369772039056],[-3.0173003355108396,38.13212802201156],[-3.016865354943445,38.13204705282209],[-3.016787753602877,38.13200073777517],[-3.0165218553109083,38.13184205744221],[-3.01655423460479,38.13174782389018],[-3.0166651825109234,38.13142493535878],[-3.0167413538987247,38.13122421474975],[-3.016754125405349,38.13119068629686],[-3.016743861764303,38.13111071874488]]]},"properties":{"dn_surface":30421.815815597707,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":22,"parcela":77,"recinto":4,"uso_sigpac":"OV","idPanel":"23:97:0:0:22:77:4"},"id":1341074656},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0219451548573466,38.119394974914655],[-3.0219422002363028,38.11951776896524],[-3.021951193180751,38.119581563634085],[-3.0219691606294603,38.11965244436428],[-3.022011063442293,38.119746950328164],[-3.0220679472315575,38.119926514174125],[-3.022094893375479,38.12001629693532],[-3.022097912537182,38.12010608304925],[-3.0220873144581795,38.12020858451106],[-3.0193557090487007,38.120265300662254],[-3.0192334204269007,38.11947463820351],[-3.019568580065285,38.11948742898852],[-3.0197226595713285,38.11949330889394],[-3.019837281265841,38.11949768340946],[-3.019930492225617,38.119501240689374],[-3.0200050945210513,38.11950408802207],[-3.020082559236589,38.11950705102501],[-3.020074322339852,38.119428551982566],[-3.0219451548573466,38.119394974914655]]]},"properties":{"dn_surface":21330.68352477426,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":22,"parcela":144,"recinto":1,"uso_sigpac":"OV","idPanel":"23:97:0:0:22:144:1"},"id":1346099366},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0297256049722794,38.149086660837355],[-3.0298643455946057,38.14925813782261],[-3.0300460124602027,38.14945763467375],[-3.030045565704737,38.149481698280994],[-3.0301174832764257,38.14951592997559],[-3.0305930875781524,38.14982702265394],[-3.0309563483867854,38.15004287679005],[-3.0309719881808452,38.15005215639559],[-3.030955846310755,38.150199426443095],[-3.030937883891188,38.15036228850781],[-3.029875353548696,38.149922094396345],[-3.029262186282109,38.14952216611189],[-3.0297256049722794,38.149086660837355]]]},"properties":{"dn_surface":7354.863500327402,"provincia":23,"municipio":48,"agregado":0,"zona":0,"poligono":12,"parcela":59,"recinto":1,"uso_sigpac":"OV","idPanel":"23:48:0:0:12:59:1"},"id":1204495212},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-2.9207468879668053,38.145793073932396],[-2.92083980304538,38.14577430517372],[-2.920962805805374,38.145727992641355],[-2.9211197795827553,38.14563195865642],[-2.9212252683550273,38.14553732361122],[-2.9213550017836685,38.145426441922126],[-2.9214219983397127,38.14536071857728],[-2.9215177465017286,38.14531247317794],[-2.9216201431897364,38.14527714764489],[-2.921734051584247,38.14523848108507],[-2.9219037021527408,38.14518024110379],[-2.9221121198639044,38.14511820076737],[-2.9222175365518055,38.1450905295888],[-2.9224313857365476,38.14504619015656],[-2.92252828976308,38.14505008271263],[-2.9226506638802983,38.145023378805746],[-2.9226991406201805,38.14500140899801],[-2.922801059539679,38.14498161345614],[-2.9229484402283097,38.144926153752294],[-2.9230925016830858,38.14487853531932],[-2.923276427471723,38.14482218713371],[-2.9231900108830637,38.14493257931963],[-2.923087251258626,38.145084630404995],[-2.922912575738881,38.14535704814162],[-2.922791542726251,38.14549526824774],[-2.9227609496160416,38.14553642423297],[-2.922755585197692,38.14556031349662],[-2.9227572624166163,38.1455940934066],[-2.922756370582066,38.145612620766485],[-2.9227192471307064,38.14564657166727],[-2.922648484585344,38.14576161497155],[-2.922376375302754,38.145931114665785],[-2.9223682406652416,38.145936181526565],[-2.9223547022144314,38.14594712074904],[-2.922262357943496,38.14602173058819],[-2.922057033154788,38.146144072854185],[-2.9218959480536553,38.146196598053535],[-2.92180435983043,38.14614588418354],[-2.921108925017501,38.14642876756514],[-2.920397028145762,38.14609679221192],[-2.92046844028866,38.14601825125999],[-2.920613873861067,38.145981850328695],[-2.9206392492362405,38.14583847702821],[-2.920727709333015,38.14582610030925],[-2.9207468879668053,38.145793073932396]]]},"properties":{"dn_surface":18721.1984537415,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":16,"parcela":178,"recinto":1,"uso_sigpac":"OV","idPanel":"23:97:0:0:16:178:1"},"id":1112590751},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.052895437260445,38.12905047196824],[-3.051843620667227,38.12898097844308],[-3.0518435804340887,38.12892492027132],[-3.0508744940682946,38.12886358569122],[-3.050785556203011,38.12846328105934],[-3.052867817211468,38.12857425662574],[-3.0528781160565086,38.128615079011155],[-3.0528815375495872,38.1287674812983],[-3.0528872925646477,38.128990992303414],[-3.0528930157284746,38.129015053396074],[-3.052895437260445,38.12905047196824]]]},"properties":{"dn_surface":8674.21520294821,"provincia":23,"municipio":48,"agregado":0,"zona":0,"poligono":20,"parcela":270,"recinto":1,"uso_sigpac":"OV","idPanel":"23:48:0:0:20:270:1"},"id":1112039892},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-2.92587740677982,38.1565678174797],[-2.925824767586575,38.15670162116075],[-2.9258101344591454,38.1567249547042],[-2.925797412405754,38.15678343776068],[-2.9258145927937087,38.15694522610352],[-2.9258373345747493,38.15698904669591],[-2.9258801024000407,38.15707143997082],[-2.9258327865538165,38.15702561526527],[-2.925807304728466,38.15698630329885],[-2.9257911259780007,38.15695989946411],[-2.925773942237284,38.156941962190245],[-2.9257525675447322,38.156929891410755],[-2.9257161054255723,38.156906253604234],[-2.9256695841838165,38.15688261663587],[-2.925625745151236,38.156866857818784],[-2.9255902888605165,38.15686057139101],[-2.9254987567960473,38.1568630021431],[-2.9254410884606,38.15687607791281],[-2.9251945733208693,38.15693399770538],[-2.924920314079989,38.156991413745544],[-2.924869518905551,38.15699987946823],[-2.924834733167126,38.15699686198292],[-2.9248020429046173,38.156989653545764],[-2.9247685136135524,38.156974146185775],[-2.924710760620877,38.15694195883748],[-2.9246302071738364,38.15686501128542],[-2.9245478088697974,38.15679108121864],[-2.924468597365534,38.156733914959204],[-2.924312687251728,38.156616145859886],[-2.924189300600547,38.156521846929024],[-2.9238826771532107,38.15626719715175],[-2.9235530872898154,38.15600349575675],[-2.923286699494671,38.15580106691573],[-2.9229900521890855,38.155586484314455],[-2.9228866999640584,38.1555195950468],[-2.9228609658434075,38.155495370507],[-2.9227369086399317,38.155390929472645],[-2.922680244456544,38.155353293886975],[-2.9226204798069886,38.155316998567685],[-2.9225670854042223,38.1552921034756],[-2.9225083265831073,38.15527726750611],[-2.9224259299554487,38.15526075431768],[-2.922352880826764,38.15525142274433],[-2.922322662387649,38.15524801298591],[-2.9222540137583923,38.15523879289191],[-2.9222143656776463,38.15523074626435],[-2.9221857515348932,38.15522193017807],[-2.922403100164419,38.15513549347284],[-2.9226214898263807,38.15519791686225],[-2.9226865057385867,38.15522267030016],[-2.922784642743149,38.15526003515035],[-2.9227984569585894,38.15526527467836],[-2.922929929632242,38.15527157954628],[-2.9230351753293076,38.155272099224305],[-2.9232387876950967,38.15527304805582],[-2.923296456030546,38.155252176277486],[-2.9233597167721217,38.15522400469922],[-2.923597951429215,38.155413881639504],[-2.9236984236321404,38.1553884450764],[-2.9237840757911844,38.155454114777115],[-2.9244695160221763,38.15597360002093],[-2.924540078240041,38.15595057409337],[-2.9247917834545243,38.15614829565723],[-2.9248982135145796,38.15609807464318],[-2.9249517185584737,38.15612821429241],[-2.9249982565640362,38.15615455945385],[-2.9254481669782493,38.156527168599666],[-2.9254890673154033,38.15660001237171],[-2.92587740677982,38.1565678174797]]]},"properties":{"dn_surface":18519.7893078827,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":15,"parcela":8,"recinto":3,"uso_sigpac":"OV","idPanel":"23:97:0:0:15:8:3"},"id":1112589849},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0154393349663513,38.109241593621164],[-3.0157700513576,38.10939378133156],[-3.015829714586122,38.10942126307913],[-3.0156058859068686,38.10980640233274],[-3.015537702473265,38.1099412403407],[-3.0152268872359023,38.11055611919201],[-3.015725981822668,38.110731630387534],[-3.015481990474134,38.11126097191189],[-3.0154324970093365,38.11132118247875],[-3.015412996510446,38.11134317491758],[-3.015452127427734,38.11136507180274],[-3.0154035073573,38.11149618070178],[-3.0153638693348395,38.111470246253646],[-3.015248978581219,38.111381243009504],[-3.0148910562070346,38.11113455520254],[-3.014765992336928,38.11103782551962],[-3.014718716723842,38.111010163561104],[-3.0146464814780045,38.110982502440805],[-3.0146784902917436,38.10971430365157],[-3.0146848546712017,38.10960164332252],[-3.0147229386887213,38.109533232739324],[-3.014773571254228,38.109470228483985],[-3.0148389249571514,38.10943354005346],[-3.01500054733831,38.10938169881743],[-3.0152388163612076,38.10929506094652],[-3.0154393349663513,38.109241593621164]],[[-3.01518893313789,38.11076260990353],[-3.0150902655588245,38.11086108134603],[-3.015337010362715,38.11094966884775],[-3.015381857738308,38.11089715119212],[-3.015328020771028,38.1108676192309],[-3.0153459538539322,38.11081838644333],[-3.01518893313789,38.11076260990353]]]},"properties":{"dn_surface":15961.23622121131,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":23,"parcela":144,"recinto":1,"uso_sigpac":"OV","idPanel":"23:97:0:0:23:144:1"},"id":1210281789}]}"""

GEOJSON_FEATURES = json.loads(GEOJSON_STR)["features"]


# ----
# OVERLAY DE PARCELAS GEOJSON SOBRE IMÁGENES DE INFERENCIA
# ----

def _geo_to_pixel(
    lon: float, lat: float,
    transform: rasterio.transform.Affine,
    img_crs: str
) -> Tuple[int, int]:
    """
    Convierte coordenadas geográficas (lon/lat WGS84) a píxel (col, row)
    usando la transformación afín y el CRS de la imagen.
    """
    transformer = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    col, row = ~transform * (x, y)
    return int(round(col)), int(round(row))


def _polygon_ring_to_pixel_mask(
    ring:      List[Tuple[float, float]],
    transform: rasterio.transform.Affine,
    img_crs:   str,
    h:         int,
    w:         int,
) -> np.ndarray:
    """
    Rasteriza un anillo de polígono (lista de [lon, lat]) a una máscara booleana (H, W).
    Usa el algoritmo de scanline (ray-casting) sobre los píxeles del anillo.
    """
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    transformer = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)
    pixel_coords = []
    for lon, lat in ring:
        x, y = transformer.transform(lon, lat)
        col, row = ~transform * (x, y)
        pixel_coords.append((float(col), float(row)))

    img_mask = _Image.new("L", (w, h), 0)
    _ImageDraw.Draw(img_mask).polygon(pixel_coords, outline=1, fill=1)
    return np.array(img_mask, dtype=bool)


def compute_parcela_metrics(
    score_map:        np.ndarray,
    valid_mask:       np.ndarray,
    meta:             dict,
    geojson_features: List[dict],
    base_name:        str,
    exp_label:        str,
    extra_bands:      List[str],
    indices:          List[str],
    mes:              int,
    anio:             int,
) -> List[Dict]:
    """
    Calcula métricas de anomalía (p50, p90, p95, p99, contrast, spread, ratio)
    para cada parcela del GeoJSON sobre el score_map de una imagen.

    Solo se consideran los píxeles que caen dentro de la parcela Y dentro de
    la máscara válida (valid_mask). Si una parcela no tiene píxeles válidos
    en la imagen, se omite.

    Returns:
        Lista de dicts con métricas por parcela.
    """
    img_crs   = str(meta["crs"])
    transform = meta["transform"]
    h, w      = score_map.shape
    rows_out: List[Dict] = []

    for feature in geojson_features:
        geom  = feature.get("geometry", {})
        props = feature.get("properties", {})
        id_panel = props.get("idPanel", str(feature.get("id", "unknown")))

        if geom.get("type") != "Polygon":
            continue

        # Solo el anillo exterior (índice 0) define el área de la parcela
        rings = geom["coordinates"]
        if not rings:
            continue

        outer_ring = rings[0]
        try:
            parcela_mask = _polygon_ring_to_pixel_mask(outer_ring, transform, img_crs, h, w)
        except Exception as e:
            logger.warning(f"   ⚠️  No se pudo rasterizar parcela {id_panel}: {e}")
            continue

        # Intersección con máscara válida
        combined_mask = parcela_mask & valid_mask
        pixel_scores  = score_map[combined_mask]

        if len(pixel_scores) < 5:
            logger.debug(f"   Parcela {id_panel}: solo {len(pixel_scores)} píxeles válidos, omitida.")
            continue

        pcts           = compute_percentiles(pixel_scores)
        contrast_score = round(pcts["p99"] - pcts["p50"], 6)
        spread         = round(pcts["p99"] - pcts["p90"], 6)
        anomaly_ratio  = round(pcts["p90"] / (float(np.median(pixel_scores)) + 1e-8), 6)

        rows_out.append({
            "experimento":    exp_label,
            "bandas_extra":   "|".join(extra_bands) if extra_bands else "RGB_only",
            "indices_activos": "|".join(indices)    if indices     else "none",
            "imagen":         base_name,
            "mes":            str(mes).zfill(2),
            "anio":           str(anio),
            "algoritmo":      "AE",
            "id_panel":       id_panel,
            "provincia":      props.get("provincia", ""),
            "municipio":      props.get("municipio", ""),
            "poligono":       props.get("poligono", ""),
            "parcela":        props.get("parcela", ""),
            "recinto":        props.get("recinto", ""),
            "dn_surface":     props.get("dn_surface", ""),
            "n_pixels_validos": int(len(pixel_scores)),
            "p50":            round(pcts["p50"], 6),
            "p90":            round(pcts["p90"], 6),
            "p95":            round(pcts["p95"], 6),
            "p99":            round(pcts["p99"], 6),
            "mean_score":     round(float(np.mean(pixel_scores)), 6),
            "std_score":      round(float(np.std(pixel_scores)),  6),
            "contrast_score": contrast_score,
            "spread":         spread,
            "anomaly_ratio":  anomaly_ratio,
        })

    return rows_out


def overlay_parcelas_on_image(
    rgb_array:   np.ndarray,
    meta:        dict,
    geojson_features: List[dict],
    outline_color: Tuple[int, int, int] = (169, 39, 245),
    line_width:  int = 2,
    label_color: Tuple[int, int, int] = (255, 255, 255),
    draw_labels: bool = False,
) -> np.ndarray:
    """
    Dibuja los polígonos del GeoJSON sobre una imagen RGB (H, W, 3) numpy array.

    Args:
        rgb_array:        Array (H, W, 3) uint8.
        meta:             Metadatos rasterio de la imagen (incluye 'crs' y 'transform').
        geojson_features: Lista de features del GeoJSON.
        outline_color:    Color del borde del polígono (R, G, B).
        line_width:       Grosor del borde en píxeles.
        label_color:      Color del texto de la etiqueta.
        draw_labels:      Si True, dibuja el idPanel como etiqueta.

    Returns:
        Array (H, W, 3) uint8 con los polígonos superpuestos.
    """
    img_crs   = str(meta["crs"])
    transform = meta["transform"]
    h, w      = rgb_array.shape[:2]

    pil_img = Image.fromarray(rgb_array, mode="RGB")
    draw    = ImageDraw.Draw(pil_img)

    # Intentar cargar fuente; fallback a fuente por defecto
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    parcelas_dibujadas = 0

    for feature in geojson_features:
        geom  = feature.get("geometry", {})
        props = feature.get("properties", {})
        label = props.get("idPanel", "")

        if geom.get("type") != "Polygon":
            continue

        for ring in geom["coordinates"]:
            pixel_coords = []
            for lon, lat in ring:
                col, row = _geo_to_pixel(lon, lat, transform, img_crs)
                pixel_coords.append((col, row))

            # Verificar si algún punto cae dentro de la imagen
            in_bounds = any(0 <= c < w and 0 <= r < h for c, r in pixel_coords)
            if not in_bounds:
                continue

            # Dibujar polígono
            if len(pixel_coords) >= 2:
                draw.line(pixel_coords + [pixel_coords[0]], fill=outline_color, width=line_width)

            # Etiqueta en el centroide aproximado
            if draw_labels and label and len(pixel_coords) >= 1:
                cx = int(np.mean([p[0] for p in pixel_coords]))
                cy = int(np.mean([p[1] for p in pixel_coords]))
                if 0 <= cx < w and 0 <= cy < h:
                    # Sombra para legibilidad
                    draw.text((cx + 1, cy + 1), label, fill=(0, 0, 0), font=font)
                    draw.text((cx, cy), label, fill=label_color, font=font)

            parcelas_dibujadas += 1

    logger.info(f"   🗺️  Parcelas superpuestas: {parcelas_dibujadas} de {len(geojson_features)}")
    return np.array(pil_img)


def save_outputs_with_parcelas(
    score_map:        np.ndarray,
    mask:             np.ndarray,
    meta:             dict,
    base_name:        str,
    alg_name:         str,
    results_dir:      str,
    geojson_features: List[dict],
) -> None:
    """
    Guarda el heatmap de anomalía con los polígonos del GeoJSON superpuestos (PNG).
    """
    os.makedirs(results_dir, exist_ok=True)
    rgb_data = colorize_heatmap_rgb(score_map, mask)
    rgb_data = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)

    rgb_overlay = overlay_parcelas_on_image(rgb_data, meta, geojson_features)

    path_out = os.path.join(
        results_dir, f"{base_name}_{alg_name}_ANOMALY_HEATMAP_PARCELAS.png"
    )
    Image.fromarray(rgb_overlay, mode="RGB").save(path_out)
    logger.info(f"💾 Heatmap con parcelas guardado: {path_out}")


def save_outputs_with_parcelas_tif(
    score_map:        np.ndarray,
    mask:             np.ndarray,
    meta:             dict,
    base_name:        str,
    alg_name:         str,
    results_dir:      str,
    geojson_features: List[dict],
) -> None:
    """
    Guarda el heatmap de anomalía con los polígonos del GeoJSON superpuestos (GeoTIFF RGB).
    """
    os.makedirs(results_dir, exist_ok=True)
    rgb_data = colorize_heatmap_rgb(score_map, mask)
    rgb_data = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)

    rgb_overlay = overlay_parcelas_on_image(rgb_data, meta, geojson_features)
    rgb_overlay = np.transpose(rgb_overlay, (2, 0, 1))  # (3, H, W)

    path_out = os.path.join(
        results_dir, f"{base_name}_{alg_name}_ANOMALY_HEATMAP_PARCELAS.tif"
    )
    out_meta = {**meta, "count": 3, "dtype": "uint8", "nodata": 0}
    with rasterio.open(path_out, "w", **out_meta) as dst:
        dst.write(rgb_overlay)
    logger.info(f"💾 GeoTIFF con parcelas guardado: {path_out}")


# ----
# CONFIGURACIÓN CENTRALIZADA
# ----

@dataclass
class PipelineConfig:
    use_extra_bands:   bool  = True
    use_indices:       bool  = True
    use_meteo:         bool  = True
    results_dir:       str   = "results2"
    max_per_image:     int   = 2000
    val_split:         float = 0.15
    epochs:            int   = 200
    batch_size:        int   = 4096
    lr:                float = 3e-4
    weight_decay:      float = 1e-2
    patience:          int   = 10
    min_delta:         float = 1e-6
    latent_dim:        int   = 8
    dropout:           float = 0.2      # reducido de 0.4 → menos regularización agresiva
    grad_clip:         float = 1.0      # gradient clipping para evitar saltos bruscos
    scheduler_T0:      int   = 30       # un solo ciclo coseno suave (antes: 10)
    scheduler_Tmult:   int   = 1
    scheduler_eta_min: float = 1e-6     # LR mínimo más bajo (antes: 1e-5)

    RGB_BANDS:       List[str] = field(default_factory=lambda: ["B02", "B03", "B04"])
    #EXTRA_BANDS_ALL: List[str] = field(default_factory=lambda: ["B08", "B05", "B06", "B07", "B8A", "B11", "B12"])
    #EXTRA_BANDS_ALL: List[str] = field(default_factory=lambda: ["B08", "B05", "B06", "B07", "B11", "B12"])
    EXTRA_BANDS_ALL: List[str] = field(default_factory=lambda:  ["B11", "B05", "B08", "B06", "B07", "B12"])

    # Ablation: si no es None, sobreescribe EXTRA_BANDS_ALL / INDICES activos
    active_extra_bands: Optional[List[str]] = None
    active_indices:     Optional[List[str]] = None

    # Rango de fechas para INFERENCIA (entrenamiento usa todos los archivos)
    # Formato: "YYYY-MM-DD" o None para sin restricción
    inference_date_start: Optional[str] = None
    inference_date_end:   Optional[str] = None

    def __post_init__(self):
        _extra = self.active_extra_bands if self.active_extra_bands is not None else self.EXTRA_BANDS_ALL
        self.BAND_ORDER = self.RGB_BANDS + (_extra if self.use_extra_bands else [])
        self.BIDX       = {name: i for i, name in enumerate(self.BAND_ORDER)}

        _ALL_INDICES = ["NDVI", "GNDVI", "MSAVI2", "NDRE", "NDMI", "EVI", "NDWI", "CRI1"]
        _INDICES_NEED_EXTRA = {
            "NDVI", "GNDVI", "MSAVI2", "NDRE", "NDMI", "EVI", "SAVI", "NDWI", "CRI1", "PSRI"
        }

        # Bandas requeridas por cada índice (según calculate_indices).
        # Un índice solo se incluye si TODAS sus bandas requeridas están en BIDX.
        _INDEX_REQUIREMENTS: Dict[str, set] = {
            "NDVI":   {"B08"},
            "GNDVI":  {"B08"},
            "MSAVI2": {"B08"},
            "EVI":    {"B08"},
            "SAVI":   {"B08"},
            "NDWI":   {"B08"},
            "NDRE":   {"B08", "B05"},
            "CRI1":   {"B08", "B05"},
            "NDMI":   {"B08", "B11"},
            "PSRI":   {"B05"},
        }

        _candidate_indices = self.active_indices if self.active_indices is not None else _ALL_INDICES

        if self.use_indices and self.use_extra_bands:
            # Solo incluir índices cuyos requerimientos de bandas están disponibles.
            # Esto garantiza que N_SPECTRAL_FEATURES coincida exactamente con lo
            # que build_feature_stack produce, evitando desajustes en el scaler.
            self.INDICES = [
                idx for idx in _candidate_indices
                if all(b in self.BIDX for b in _INDEX_REQUIREMENTS.get(idx, set()))
            ]
            omitted = [idx for idx in _candidate_indices if idx not in self.INDICES]
            if omitted:
                logger.warning(
                    f"Índices omitidos por falta de bandas requeridas: {omitted} "
                    f"(bandas activas: {list(self.BIDX.keys())})"
                )
        elif self.use_indices and not self.use_extra_bands:
            self.INDICES = [i for i in _candidate_indices if i not in _INDICES_NEED_EXTRA]
            if not self.INDICES:
                logger.warning(
                    "use_indices=True pero use_extra_bands=False: "
                    "ningún índice calculable. Se desactivan."
                )
        else:
            self.INDICES = []

        # FEATURE_NAMES y N_SPECTRAL_FEATURES derivados de la misma fuente
        self.FEATURE_NAMES       = self.BAND_ORDER + self.INDICES
        self.N_SPECTRAL_FEATURES = len(self.FEATURE_NAMES)
        self.N_TEMPORAL_FEATURES = 6

        self.METEO_COLS = [
            "TMax", "TMin", "TMed",
            "HumMax", "HumMin", "HumMed",
            "VelViento", "VelVientoMax", "DirViento",
            "Rad", "Precip", "ETo"
        ]
        self.N_METEO_FEATURES = len(self.METEO_COLS) if self.use_meteo else 0

        _all_weights = {
            "B05": 1.0, "B06": 1.0, "B07": 1.0, "B11": 1.0, "B12": 1.0,
            # Índices más sensibles al algodoncillo
            "NDRE":   1.0,   # red-edge: primer indicador de estrés en hoja joven
            "CRI1":   1.0,   # contenido de clorofila en brotes
            "NDVI":   1.0,   # vigor general
            "GNDVI":  1.0,   # clorofila verde
            "NDMI":   1.0,   # estrés hídrico
            "MSAVI2": 1.0,   # vegetación con suelo (brotes jóvenes)
            # Bandas visibles: la cera blanca sube B02/B03/B04
            "B02":    1.0,
            "B03":    1.0,
            "B04":    1.0,
            # Menos relevantes para esta plaga
            "EVI":    1.0,
            "NDWI":   1.0,
        }
        """
        _all_weights = {
            "B05": 1.0, "B06": 1.0, "B07": 1.0, "B11": 1.0, "B12": 1.0,
            # Índices más sensibles al algodoncillo
            "NDRE":   3.0,   # red-edge: primer indicador de estrés en hoja joven
            "CRI1":   3.0,   # contenido de clorofila en brotes
            "NDVI":   2.5,   # vigor general
            "GNDVI":  2.5,   # clorofila verde
            "NDMI":   2.0,   # estrés hídrico
            "MSAVI2": 2.0,   # vegetación con suelo (brotes jóvenes)
            # Bandas visibles: la cera blanca sube B02/B03/B04
            "B02":    2.0,
            "B03":    2.0,
            "B04":    1.5,
            # Menos relevantes para esta plaga
            "EVI":    1.0,
            "NDWI":   0.5,
        }
        """
        self.FEATURE_WEIGHTS = {k: v for k, v in _all_weights.items() if k in self.FEATURE_NAMES}


# --- FENOLOGÍA DEL OLIVO (hemisferio norte) ---
OLIVE_PHENOLOGY = {
    1: "dormancia",     2: "dormancia",
    3: "brotacion",     4: "floracion",
    5: "floracion",     6: "cuajado",
    7: "engorde_fruto", 8: "engorde_fruto",
    9: "maduracion",    10: "maduracion",
    11: "cosecha",      12: "post_cosecha"
}
PHENOLOGY_ENCODING = {
    "dormancia": 0, "brotacion": 1, "floracion": 2,
    "cuajado": 3,   "engorde_fruto": 4, "maduracion": 5,
    "post_cosecha": 6, "cosecha": 7
}

INVALID_SCL = [3, 8, 9, 10, 11]


# ----
# CARGA Y PREPROCESADO DEL CSV METEOROLÓGICO
# ----

def load_meteo_csv(csv_path: str, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Carga el CSV de Villacarrillo, limpia valores n/d y --:--,
    selecciona las columnas numéricas útiles y parsea la fecha.
    """
    df = pd.read_csv(csv_path, sep=";", decimal=".", encoding="utf-8")
    df.columns = [c.replace("Ja102", "") for c in df.columns]

    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["FECHA"])
    df = df.set_index("FECHA").sort_index()

    df.replace(["n/d", "N/D", "--:--", ""], np.nan, inplace=True)

    available = [c for c in cfg.METEO_COLS if c in df.columns]
    missing   = [c for c in cfg.METEO_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Columnas meteorológicas no encontradas en CSV: {missing}")

    df = df[available].copy()

    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")
    df = df.interpolate(method="time").ffill().bfill()
    df = df[~df.index.duplicated(keep='first')]

    logger.info(
        f"📋 CSV meteorológico cargado: {len(df)} días válidos | "
        f"Columnas: {list(df.columns)}"
    )
    return df


def get_meteo_vector(
    fecha:        pd.Timestamp,
    meteo_df:     Optional[pd.DataFrame],
    meteo_scaler: Optional[RobustScaler]
) -> Optional[np.ndarray]:
    """
    Obtiene el vector meteorológico escalado para una fecha dada.
    Retorna None si meteo_df o meteo_scaler son None, o si la fecha
    más cercana está a más de 2 días.
    """
    if meteo_df is None or meteo_scaler is None:
        return None
    try:
        idx          = meteo_df.index.get_indexer([fecha], method='nearest')[0]
        closest_date = meteo_df.index[idx]

        if abs((closest_date - fecha).days) > 2:
            return None

        row = meteo_df.iloc[idx].values.astype(np.float32)
        return meteo_scaler.transform(row.reshape(1, -1))[0]
    except Exception as e:
        logger.warning(f"Error obteniendo vector meteo para {fecha}: {e}")
        return None


# ----
# MODELO: Autoencoder con rama meteorológica
# ----

class OliveAnomalyAutoencoder(nn.Module):
    """
    Autoencoder con BatchNorm y Dropout.

    Arquitectura:
        Encoder espectral:  input_dim → 64 → 32 → 16 → latent_dim
        Rama meteo:         meteo_dim → 16 → latent_dim   (si meteo_dim > 0)
        Fusión:             concat([z_spec, z_meteo]) → latent_dim
        Decoder:            latent_dim → 16 → 32 → 64 → input_dim
    """

    def __init__(
        self,
        input_dim:  int,
        latent_dim: int   = 8,
        dropout:    float = 0.2,
        meteo_dim:  int   = 0
    ):
        super().__init__()
        self.use_meteo = meteo_dim > 0

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

        if self.use_meteo:
            self.meteo_branch = nn.Sequential(
                nn.Linear(meteo_dim, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(16, latent_dim)
            )
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU()
            )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 32),         nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 64),         nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, input_dim)
        )

    def encode(
        self,
        x:       torch.Tensor,
        x_meteo: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Representación latente fusionada."""
        z = self.encoder(x)
        if self.use_meteo and x_meteo is not None:
            z_m = self.meteo_branch(x_meteo)
            z   = self.fusion(torch.cat([z, z_m], dim=1))
        return z

    def forward(
        self,
        x:       torch.Tensor,
        x_meteo: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder(self.encode(x, x_meteo))


# ----
# EARLY STOPPING con restauración de mejores pesos
# ----

class EarlyStopping:
    """Detiene el entrenamiento y restaura los pesos del mejor modelo."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience     = patience
        self.min_delta    = min_delta
        self.counter      = 0
        self.best_loss:    Optional[float] = None
        self.early_stop   = False
        self.best_weights: Optional[Dict]  = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss    = val_loss
            self.counter      = 0
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("✅ Pesos del mejor modelo restaurados.")


# ----
# ENTRENAMIENTO
# ----

def _batch_loss_meteo(
    model:     nn.Module,
    x:         torch.Tensor,
    x_meteo:   Optional[torch.Tensor],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    grad_clip: float = 1.0
) -> float:
    """Paso de entrenamiento con gradient clipping para evitar saltos bruscos."""
    optimizer.zero_grad()
    out  = model(x, x_meteo)
    loss = criterion(out, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def train_global_ae(
    X_train:       np.ndarray,
    X_val:         np.ndarray,
    X_meteo_train: Optional[np.ndarray] = None,
    X_meteo_val:   Optional[np.ndarray] = None,
    cfg:           Optional[PipelineConfig] = None
) -> nn.Module:
    """
    Entrena el autoencoder con:
      - Split por imagen (val separado, sin data leakage)
      - Scheduler coseno suave (T0=30)
      - Gradient clipping
      - Dropout reducido (0.2)
      - Early stopping con paciencia 20

    Args:
        X_train:       Array (n_train, features).
        X_val:         Array (n_val, features).
        X_meteo_train: Array (n_train, meteo_features) o None.
        X_meteo_val:   Array (n_val,   meteo_features) o None.
        cfg:           Configuración del pipeline.

    Returns:
        Modelo entrenado en modo eval con los mejores pesos restaurados.
    """
    if cfg is None:
        cfg = PipelineConfig()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    input_dim = X_train.shape[1]
    meteo_dim = X_meteo_train.shape[1] if X_meteo_train is not None else 0
    model     = OliveAnomalyAutoencoder(
        input_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        meteo_dim=meteo_dim
    ).to(device)

    logger.info(
        f"🏗️  Modelo | input_dim={input_dim} | meteo_dim={meteo_dim} | "
        f"rama_meteo={'✅' if meteo_dim > 0 else '❌'} | "
        f"dropout={cfg.dropout} | grad_clip={cfg.grad_clip}"
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Scheduler coseno suave: un ciclo de T0 épocas sin reinicios bruscos
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.scheduler_T0,
        T_mult=cfg.scheduler_Tmult,
        eta_min=cfg.scheduler_eta_min
    )

    early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    criterion      = nn.MSELoss()

    # Construir datasets separados (sin random_split → sin data leakage entre imágenes)
    X_t = torch.from_numpy(X_train.astype(np.float32))
    X_v = torch.from_numpy(X_val.astype(np.float32))

    if X_meteo_train is not None and X_meteo_val is not None:
        train_ds = TensorDataset(X_t, torch.from_numpy(X_meteo_train.astype(np.float32)))
        val_ds   = TensorDataset(X_v, torch.from_numpy(X_meteo_val.astype(np.float32)))
    else:
        train_ds = TensorDataset(X_t)
        val_ds   = TensorDataset(X_v)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

    logger.info(f"🚀 Entrenando: {len(X_train)} muestras train | {len(X_val)} val")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_b        = batch[0].to(device)
            m_b        = batch[1].to(device) if len(batch) > 1 else None
            train_loss += _batch_loss_meteo(model, x_b, m_b, criterion, optimizer, cfg.grad_clip)
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_b      = batch[0].to(device)
                m_b      = batch[1].to(device) if len(batch) > 1 else None
                val_loss += criterion(model(x_b, m_b), x_b).item()
        val_loss /= len(val_loader)

        scheduler.step()

        #if epoch % 5 == 0:
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"🛑 Early stopping en época {epoch}")
            break

    early_stopping.restore_best_weights(model)
    return model.eval()


# ----
# SCORE DE ANOMALÍA PONDERADO
# ----

def compute_weighted_anomaly_score(
    original:      torch.Tensor,
    reconstructed: torch.Tensor,
    cfg:           PipelineConfig
) -> torch.Tensor:
    """MSE ponderado por importancia de cada feature para olivos."""
    n_total = original.shape[1]
    n_spec  = cfg.N_SPECTRAL_FEATURES
    n_temp  = n_total - n_spec

    spec_w  = [cfg.FEATURE_WEIGHTS.get(f, 1.0) for f in cfg.FEATURE_NAMES]
    temp_w  = [1.0] * n_temp
    weights = torch.tensor(spec_w + temp_w, dtype=torch.float32, device=original.device)
    weights = weights / weights.sum()

    return torch.sum(weights * (original - reconstructed) ** 2, dim=1)


# ----
# FUNCIONES DE SOPORTE
# ----

def _extract_date_str(base_name: str) -> Optional[str]:
    """
    Busca el primer segmento con formato YYYY-MM-DD o YYYYMMDD en el nombre,
    independientemente del prefijo del mapa (AUR1, AMO1, etc.).
    """
    import re as _re
    for part in base_name.split("_"):
        if _re.match(r'^\d{4}-\d{2}-\d{2}$', part):
            return part
        if _re.match(r'^\d{8}$', part):
            return part
    return None


def get_date_info(base_name: str) -> Tuple[int, int]:
    """Extrae (mes, año) del nombre de archivo. Soporta cualquier prefijo de mapa."""
    date_str = _extract_date_str(base_name)
    if date_str:
        try:
            if '-' in date_str:
                parts = date_str.split("-")
                return int(parts[1]), int(parts[0])
            else:
                return int(date_str[4:6]), int(date_str[:4])
        except (IndexError, ValueError) as e:
            logger.warning(f"No se pudo parsear fecha de '{base_name}': {e}. Usando (1, 2024).")
    else:
        logger.warning(f"No se encontró fecha en '{base_name}'. Usando (1, 2024).")
    return 1, 2024


def get_image_date(base_name: str) -> Optional[pd.Timestamp]:
    """
    Extrae la fecha completa del nombre de archivo como Timestamp.
    Soporta formatos YYYY-MM-DD y YYYYMMDD con cualquier prefijo de mapa.
    """
    date_str = _extract_date_str(base_name)
    if date_str:
        try:
            fmt = "%Y-%m-%d" if '-' in date_str else "%Y%m%d"
            return pd.to_datetime(date_str, format=fmt)
        except Exception:
            pass
    logger.warning(f"No se pudo extraer fecha completa de '{base_name}'.")
    return None


def _find_mask_path(f_path: str) -> Optional[str]:
    """
    Detecta el archivo de máscara según el prefijo del mapa:
      - PREFIX_DATE_SCL_dataMask.tif  (e.g. AUR1_2015-12-06_SCL_dataMask.tif)
      - PREFIX_DATE_dataMask.tif      (e.g. AMO1_2015-12-06_dataMask.tif)
    """
    candidate1 = f_path.replace("_stack.tif", "_SCL_dataMask.tif")
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = f_path.replace("_stack.tif", "_dataMask.tif")
    if os.path.exists(candidate2):
        return candidate2
    return None


def load_mask(f_path: str, h: int, w: int) -> np.ndarray:
    """
    Carga máscara SCL + DataMask. Detecta automáticamente el patrón de nombre.
    Soporta máscaras con 1 banda (solo dataMask) o 2 bandas (SCL + dataMask).
    """
    mask_path = _find_mask_path(f_path)
    if mask_path:
        with rasterio.open(mask_path) as src:
            n_bands = src.count
            scl     = src.read(1)
            dm      = src.read(2) if n_bands >= 2 else src.read(1)
        if n_bands >= 2:
            return (dm > 0) & (~np.isin(scl, INVALID_SCL))
        else:
            return dm > 0
    logger.warning(f"Máscara no encontrada: {os.path.basename(f_path)}")
    return np.ones((h, w), dtype=bool)


def calculate_indices(data: np.ndarray, cfg: PipelineConfig, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """Calcula índices espectrales activos según la configuración."""
    if not cfg.INDICES:
        return {}

    red, green, blue = data[cfg.BIDX["B04"]], data[cfg.BIDX["B03"]], data[cfg.BIDX["B02"]]
    nir  = data[cfg.BIDX["B08"]] if "B08" in cfg.BIDX else None
    re   = data[cfg.BIDX["B05"]] if "B05" in cfg.BIDX else None
    swir = data[cfg.BIDX["B11"]] if "B11" in cfg.BIDX else None

    all_possible: Dict[str, np.ndarray] = {}

    if nir is not None:
        t     = 2.0 * nir + 1.0
        inner = t ** 2 - 8.0 * (nir - red)
        all_possible.update({
            "NDVI":   (nir - red)   / (nir + red   + eps),
            "GNDVI":  (nir - green) / (nir + green + eps),
            "MSAVI2": (t - np.sqrt(np.maximum(inner, 0) + eps)) / 2.0,
            "EVI":    2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps),
            "SAVI":   1.5 * (nir - red) / (nir + red + 0.5 + eps),
            "NDWI":   (green - nir) / (green + nir + eps),
        })
    if nir is not None and re is not None:
        all_possible.update({
            "NDRE": (nir - re) / (nir + re + eps),
            "CRI1": (1 / (green + eps)) - (1 / (re + eps)),
        })
    if nir is not None and swir is not None:
        all_possible["NDMI"] = (nir - swir) / (nir + swir + eps)
    if re is not None:
        all_possible["PSRI"] = (red - blue) / (re + eps)

    idx = {k: all_possible[k] for k in cfg.INDICES if k in all_possible}
    return {k: np.clip(v, -1.0, 1.0) for k, v in idx.items()}


def build_feature_stack(img: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Construye la matriz de features (H*W, N_SPECTRAL_FEATURES)."""
    indices = calculate_indices(img, cfg)
    bands   = [img[cfg.BIDX[b]] for b in cfg.BAND_ORDER]
    idxs    = [indices[i] for i in cfg.INDICES if i in indices]
    layers  = bands + idxs
    return np.stack(layers, axis=0).reshape(len(layers), -1).T


def add_temporal_features(X: np.ndarray, mes: int, anio: int) -> np.ndarray:
    """
    Agrega 6 features temporales:
      sin/cos anual, sin/cos semestral, fenología normalizada, año normalizado.
    """
    n     = X.shape[0]
    phase = PHENOLOGY_ENCODING[OLIVE_PHENOLOGY[mes]]
    return np.hstack([
        X,
        np.full((n, 1), np.sin(2 * np.pi * mes / 12.0)),
        np.full((n, 1), np.cos(2 * np.pi * mes / 12.0)),
        np.full((n, 1), np.sin(2 * np.pi * mes / 6.0)),
        np.full((n, 1), np.cos(2 * np.pi * mes / 6.0)),
        np.full((n, 1), phase / 7.0),
        np.full((n, 1), (anio - 2015) / 15.0),
    ])


def build_meteo_tensor(
    n_pixels:     int,
    meteo_vector: Optional[np.ndarray],
    device:       torch.device
) -> Optional[torch.Tensor]:
    """
    Crea tensor meteorológico con broadcasting (sin np.tile → menos memoria).
    Retorna tensor (n_pixels, N_METEO_FEATURES) o None.
    """
    if meteo_vector is None:
        return None
    t = torch.from_numpy(meteo_vector.astype(np.float32)).unsqueeze(0).to(device)
    return t.expand(n_pixels, -1)


# ----
# NORMALIZACIÓN Y PERCENTILES
# ----

def normalize_scores(scores: np.ndarray, p_lo: float = 1, p_hi: float = 99) -> np.ndarray:
    """Normaliza scores al rango [0, 1] usando percentiles para robustez."""
    vmin = np.percentile(scores, p_lo)
    vmax = np.percentile(scores, p_hi)
    if vmax <= vmin:
        return np.zeros_like(scores, dtype=np.float32)
    return np.clip((scores - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def compute_percentiles(scores: np.ndarray) -> Dict[str, float]:
    """Calcula percentiles clave una sola vez para reutilizar en plots y logs."""
    return {
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


# ----
# GUARDADO DE RESULTADOS
# ----

def colorize_heatmap_rgb(score_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Convierte mapa de scores a imagen RGB con colormap Jet."""
    H, W = score_map.shape
    rgb  = np.zeros((3, H, W), dtype=np.uint8)
    mask = valid_mask > 0
    if not mask.any():
        return rgb

    vals       = score_map[mask]
    vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
    norm       = np.clip((score_map - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)

    cmap    = plt.get_cmap("jet")
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    for c in range(3):
        channel        = colored[:, :, c]
        channel[~mask] = 0
        rgb[c]         = channel
    return rgb


def save_scores_csv(
    scores_valid: np.ndarray, base_name: str,
    alg_name: str, results_dir: str
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path_csv = os.path.join(results_dir, f"{base_name}_{alg_name}_scores.csv")
    pd.DataFrame({"anomaly_score": scores_valid}).to_csv(path_csv, index=False)
    logger.info(f"   💾 CSV guardado: {path_csv}")


def save_percentile_plot(
    scores_valid: np.ndarray, base_name: str, alg_name: str,
    results_dir: str, pcts: Optional[Dict[str, float]] = None
) -> None:
    if pcts is None:
        pcts = compute_percentiles(scores_valid)

    percentiles = {
        "Mediana (P50)": pcts["p50"],
        "P90":           pcts["p90"],
        "P95":           pcts["p95"],
        "P99":           pcts["p99"],
    }
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(percentiles.keys()), list(percentiles.values()),
                  color=colors, edgecolor="black", width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title(f"Percentiles de Score de Anomalía\n{base_name} — {alg_name}", fontsize=11)
    ax.set_ylabel("Score de Anomalía")
    ax.set_xlabel("Estadístico")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path_plot = os.path.join(results_dir, f"{base_name}_{alg_name}_percentiles.png")
    fig.savefig(path_plot, dpi=150)
    plt.close(fig)
    logger.info(f"   📊 Gráfica de percentiles guardada: {path_plot}")


def save_boxplot(
    scores_valid: np.ndarray, base_name: str, alg_name: str,
    results_dir: str, pcts: Optional[Dict[str, float]] = None
) -> None:
    if pcts is None:
        pcts = compute_percentiles(scores_valid)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.boxplot(
        scores_valid, vert=True, patch_artist=True,
        boxprops=dict(facecolor="#AEC6E8", color="#2c5f8a"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="#2c5f8a"),
        capprops=dict(color="#2c5f8a"),
        flierprops=dict(marker=".", color="gray", alpha=0.3, markersize=3),
        whis=[5, 95],
    )
    for val, label, color in [
        (pcts["p50"], "Mediana (P50)", "red"),
        (pcts["p90"], "P90",          "#FF7F0E"),
        (pcts["p95"], "P95",          "#2CA02C"),
        (pcts["p99"], "P99",          "#9467BD"),
    ]:
        ax.axhline(val, linestyle="--", color=color, linewidth=1.4,
                   label=f"{label}: {val:.4f}")

    
    ax.set_title(f"Boxplot de Score de Anomalía\n{base_name} — {alg_name}", fontsize=11)
    ax.set_ylabel("Score de Anomalía")
    ax.set_xticks([])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path_box = os.path.join(results_dir, f"{base_name}_{alg_name}_boxplot.png")
    fig.savefig(path_box, dpi=150)
    plt.close(fig)
    logger.info(f"   📦 Boxplot guardado: {path_box}")
    


def save_percentile_heatmaps_tif(
    scores_valid: np.ndarray, valid_idx: np.ndarray,
    h: int, w: int, mask: np.ndarray, meta: dict,
    base_name: str, alg_name: str, results_dir: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    for pct, pct_label in [(50, "P50"), (90, "P90"), (95, "P95"), (99, "P99")]:
        threshold   = np.percentile(scores_valid, pct)
        binary_flat = np.zeros(h * w, dtype=np.float32)
        binary_flat[valid_idx] = (scores_valid >= threshold).astype(np.float32)
        binary_map  = binary_flat.reshape(h, w)

        path_raw = os.path.join(results_dir, f"{base_name}_{alg_name}_{pct_label}_HEATMAP.tif")
        meta_raw = {**meta, "count": 1, "dtype": "float32", "nodata": -1}
        with rasterio.open(path_raw, "w", **meta_raw) as dst:
            out        = binary_map.copy()
            out[~mask] = -1
            dst.write(out.astype("float32"), 1)

        path_rgb = os.path.join(results_dir, f"{base_name}_{alg_name}_{pct_label}_HEATMAP_RGB.tif")
        rgb_data = colorize_heatmap_rgb(binary_map, mask)
        meta_rgb = {**meta, "count": 3, "dtype": "uint8", "nodata": 0}
        with rasterio.open(path_rgb, "w", **meta_rgb) as dst:
            dst.write(rgb_data)

        logger.info(f"   🗺️  Heatmap {pct_label} guardado (umbral={threshold:.4f}): {path_raw}")


def save_percentile_heatmaps(
    scores_valid: np.ndarray, valid_idx: np.ndarray,
    h: int, w: int, mask: np.ndarray, meta: dict,
    base_name: str, alg_name: str, results_dir: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    for pct, pct_label in [(50, "P50"), (90, "P90"), (95, "P95"), (99, "P99")]:
        threshold   = np.percentile(scores_valid, pct)
        binary_flat = np.zeros(h * w, dtype=np.float32)
        binary_flat[valid_idx] = (scores_valid >= threshold).astype(np.float32)
        binary_map  = binary_flat.reshape(h, w)

        path_rgb = os.path.join(
            results_dir, f"{base_name}_{alg_name}_{pct_label}_HEATMAP_RGB.png"
        )
        rgb_data = colorize_heatmap_rgb(binary_map, mask)
        rgb_data = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(rgb_data, mode="RGB").save(path_rgb)
        logger.info(f"🗺️ Heatmap {pct_label} guardado (umbral={threshold:.4f}): {path_rgb}")


def save_outputs_tif(
    score_map: np.ndarray, mask: np.ndarray,
    meta: dict, base_name: str, alg_name: str, results_dir: str
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path_rgb = os.path.join(results_dir, f"{base_name}_{alg_name}_ANOMALY_HEATMAP_RGB.tif")
    rgb_data = colorize_heatmap_rgb(score_map, mask)
    out_meta = {**meta, "count": 3, "dtype": "uint8", "nodata": 0}
    with rasterio.open(path_rgb, "w", **out_meta) as dst:
        dst.write(rgb_data)
    logger.info(f"💾 Guardado: {path_rgb}")


def save_outputs(
    score_map: np.ndarray, mask: np.ndarray,
    meta: dict, base_name: str, alg_name: str, results_dir: str
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path_rgb = os.path.join(results_dir, f"{base_name}_{alg_name}_ANOMALY_HEATMAP_RGB.png")
    rgb_data = colorize_heatmap_rgb(score_map, mask)
    rgb_data = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb_data, mode="RGB").save(path_rgb)
    logger.info(f"💾 Guardado: {path_rgb}")


# ----
# RECOLECCIÓN DE MUESTRAS
# ----

def collect_training_samples(
    files:        List[str],
    meteo_df:     Optional[pd.DataFrame],
    meteo_scaler: Optional[RobustScaler],
    cfg:          PipelineConfig
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Recolecta muestras de entrenamiento de una lista de archivos.
    Retorna (X_spectral_temporal, X_meteo).
    """
    all_samples = []
    all_meteo   = [] if cfg.use_meteo and meteo_df is not None else None

    for f_path in files:
        base_name = os.path.basename(f_path).replace("_stack.tif", "")
        mes, anio = get_date_info(base_name)
        fecha     = get_image_date(base_name)

        with rasterio.open(f_path) as src:
            img  = src.read().astype("float32")
            h, w = img.shape[1], img.shape[2]

        mask_2d   = load_mask(f_path, h, w)
        valid_idx = np.where(mask_2d.flatten())[0]
        X_all     = build_feature_stack(img, cfg)
        X_valid   = X_all[valid_idx]

        if len(X_valid) == 0:
            logger.warning(f"Sin píxeles válidos en {base_name}")
            continue

        n        = min(len(X_valid), cfg.max_per_image)
        idx_r    = np.random.choice(len(X_valid), n, replace=False)
        X_sample = add_temporal_features(X_valid[idx_r], mes, anio)
        all_samples.append(X_sample)

        if all_meteo is not None and fecha is not None:
            m_vec = get_meteo_vector(fecha, meteo_df, meteo_scaler)
            if m_vec is not None:
                all_meteo.append(np.tile(m_vec, (n, 1)))
            else:
                all_meteo.append(np.zeros((n, cfg.N_METEO_FEATURES), dtype=np.float32))

    if not all_samples:
        raise ValueError("No se encontraron muestras válidas.")

    X_out = np.vstack(all_samples)
    M_out = np.vstack(all_meteo) if all_meteo else None
    return X_out, M_out


def collect_training_samples_split(
    files:        List[str],
    meteo_df:     Optional[pd.DataFrame],
    meteo_scaler: Optional[RobustScaler],
    cfg:          PipelineConfig
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Split por FECHA ÚNICA para evitar data leakage temporal entre train y val.

    Agrupa todos los archivos por fecha (independientemente del prefijo del mapa:
    AUR1, AMO1, etc.) y hace el split a nivel de fecha. Esto garantiza que todos
    los mapas de una misma fecha vayan juntos al mismo conjunto (train o val),
    evitando que el modelo vea la misma fecha en ambos sets.

    Ejemplo:
        AUR1_2015-12-06_stack.tif  \
        AMO1_2015-12-06_stack.tif   --> misma fecha → van juntos a train o val
        AUR1_2016-03-15_stack.tif  --> fecha distinta → puede ir al otro conjunto

    Retorna (X_train, M_train, X_val, M_val).
    """
    from collections import defaultdict

    # 1. Agrupar archivos por fecha
    date_groups: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        base   = os.path.basename(f).replace("_stack.tif", "")
        # Extraer fecha usando regex: YYYY-MM-DD o YYYYMMDD
        date_str = None
        for part in base.split("_"):
            import re as _re
            if _re.match(r'^\d{4}-\d{2}-\d{2}$', part) or _re.match(r'^\d{8}$', part):
                date_str = part
                break
        key = date_str if date_str else base  # fallback: usar nombre completo
        date_groups[key].append(f)

    fechas = sorted(date_groups.keys())
    rng    = np.random.default_rng(42)
    rng.shuffle(fechas)

    n_val_fechas = max(1, int(len(fechas) * cfg.val_split))
    val_fechas   = set(fechas[:n_val_fechas])
    train_fechas = set(fechas[n_val_fechas:])

    train_files = [f for fecha in train_fechas for f in date_groups[fecha]]
    val_files   = [f for fecha in val_fechas   for f in date_groups[fecha]]

    logger.info(
        f"📂 Split por FECHA: {len(train_fechas)} fechas train "
        f"({len(train_files)} archivos) | "
        f"{len(val_fechas)} fechas val ({len(val_files)} archivos)"
    )
    if val_fechas:
        logger.info(f"   📅 Fechas en val: {sorted(val_fechas)}")

    X_tr, M_tr = collect_training_samples(train_files, meteo_df, meteo_scaler, cfg)
    X_va, M_va = collect_training_samples(val_files,   meteo_df, meteo_scaler, cfg)
    return X_tr, M_tr, X_va, M_va


# ----
# PIPELINE PRINCIPAL
# ----

def _make_experiment_folder(base_results_dir: str, extra_bands: List[str], indices: List[str]) -> str:
    """Crea subcarpeta nombrada según bandas e índices activos."""
    import hashlib
    rgb_bands   = ["B02", "B03", "B04"]
    bands_part  = "_".join(rgb_bands + extra_bands) if extra_bands else "_".join(rgb_bands)
    idx_part    = "_".join(indices) if indices else "none"
    folder_name = f"bands_{bands_part}__idx_{idx_part}"
    if len(folder_name) > 120:
        short_hash  = hashlib.md5(folder_name.encode()).hexdigest()[:8]
        folder_name = folder_name[:100] + f"__h{short_hash}"
    full_path = os.path.join(base_results_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def run_anomaly_pipeline(
    data_dir:       str,
    meteo_csv_path: Optional[str] = None,
    cfg:            Optional[PipelineConfig] = None,
    save_images:    bool = True,
    exp_label:      str  = "BASE",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Pipeline completo: recolección → escalado → entrenamiento → inferencia → guardado.

    Args:
        data_dir:       Directorio con archivos *_stack.tif.
        meteo_csv_path: Ruta al CSV meteorológico (opcional).
        cfg:            Configuración del pipeline. Si None, usa valores por defecto.
        save_images:    Si True, guarda imágenes/plots en la carpeta del experimento.
        exp_label:      Etiqueta del experimento (para el CSV comparativo).

    Returns:
        Tupla (summary_rows, parcela_rows):
          - summary_rows: métricas globales por imagen.
          - parcela_rows: métricas por parcela individual por imagen.
    """
    if cfg is None:
        cfg = PipelineConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files  = sorted(glob.glob(os.path.join(data_dir, "*_stack.tif")))

    if not files:
        logger.error(f"No se encontraron archivos en: {data_dir}")
        return []

    # --- Cargar y escalar CSV meteorológico ---
    meteo_df     = None
    meteo_scaler = None
    if cfg.use_meteo and meteo_csv_path and os.path.exists(meteo_csv_path):
        meteo_df     = load_meteo_csv(meteo_csv_path, cfg)
        meteo_scaler = RobustScaler()
        meteo_scaler.fit(meteo_df.values)
        logger.info(f"🌡️  Scaler meteorológico ajustado sobre {len(meteo_df)} días.")
    elif cfg.use_meteo and meteo_csv_path:
        logger.warning(f"CSV meteorológico no encontrado: {meteo_csv_path}. Se omite rama meteo.")

    # Carpeta del experimento nombrada según bandas e índices activos
    extra_bands = cfg.active_extra_bands if cfg.active_extra_bands is not None else cfg.EXTRA_BANDS_ALL
    indices     = cfg.active_indices     if cfg.active_indices     is not None else cfg.INDICES
    exp_folder  = _make_experiment_folder(cfg.results_dir, extra_bands, indices)

    logger.info(
        f"⚙️  [{exp_label}] Bandas extra: {'✅' if cfg.use_extra_bands else '❌'} | "
        f"Índices: {'✅' if cfg.use_indices else '❌'} | "
        f"Meteo: {'✅' if meteo_df is not None else '❌'} | "
        f"Features espectrales: {cfg.N_SPECTRAL_FEATURES} | "
        f"Carpeta: {exp_folder}"
    )
    logger.info(f"📂 Procesando {len(files)} archivos...")

    # 1. Recolección con split por fecha (evita data leakage)
    X_tr, M_tr, X_va, M_va = collect_training_samples_split(
        files, meteo_df, meteo_scaler, cfg
    )

    # 2. Escalado robusto: fit en train, transform en val
    scaler    = RobustScaler()
    X_tr_spec = scaler.fit_transform(
        np.nan_to_num(X_tr[:, :cfg.N_SPECTRAL_FEATURES], nan=0.0, posinf=0.0, neginf=0.0)
    )
    X_va_spec = scaler.transform(
        np.nan_to_num(X_va[:, :cfg.N_SPECTRAL_FEATURES], nan=0.0, posinf=0.0, neginf=0.0)
    )
    X_train_final = np.hstack([X_tr_spec, X_tr[:, cfg.N_SPECTRAL_FEATURES:]])
    X_val_final   = np.hstack([X_va_spec, X_va[:, cfg.N_SPECTRAL_FEATURES:]])

    # 3. Entrenamiento con val separado
    model = train_global_ae(X_train_final, X_val_final, M_tr, M_va, cfg=cfg)
    
    
    # ── Guardar modelo entrenado ────
    model_path = os.path.join(exp_folder, "ae_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"💾 Modelo guardado: {model_path}")

    # ── Guardar scaler espectral ────
    scaler_path = os.path.join(exp_folder, "spectral_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"💾 Scaler guardado: {scaler_path}")

    # ── Guardar scaler meteorológico (si existe) ────
    if meteo_scaler is not None:
        meteo_scaler_path = os.path.join(exp_folder, "meteo_scaler.pkl")
        with open(meteo_scaler_path, "wb") as f:
            pickle.dump(meteo_scaler, f)
        logger.info(f"💾 Scaler meteo guardado: {meteo_scaler_path}")
        

    summary_rows: List[Dict] = []
    parcela_rows: List[Dict] = []

    # 4. Filtro de fechas para inferencia
    infer_start = pd.to_datetime(cfg.inference_date_start) if cfg.inference_date_start else None
    infer_end   = pd.to_datetime(cfg.inference_date_end)   if cfg.inference_date_end   else None

    if infer_start or infer_end:
        logger.info(
            f"📅 Rango de inferencia: "
            f"{infer_start.date() if infer_start else '(sin límite)'} → "
            f"{infer_end.date()   if infer_end   else '(sin límite)'}"
        )

    infer_files = []
    for f_path in files:
        base_name = os.path.basename(f_path).replace("_stack.tif", "")
        fecha     = get_image_date(base_name)
        if fecha is None:
            infer_files.append(f_path)   # sin fecha → incluir siempre
            continue
        if infer_start and fecha < infer_start:
            continue
        if infer_end   and fecha > infer_end:
            continue
        infer_files.append(f_path)

    logger.info(
        f"🔍 Imágenes para inferencia: {len(infer_files)} de {len(files)} "
        f"(filtro de fechas {'activo' if infer_start or infer_end else 'desactivado'})"
    )

    # 5. Inferencia imagen por imagen
    for f_path in infer_files:
        base_name = os.path.basename(f_path).replace("_stack.tif", "")
        mes, anio = get_date_info(base_name)
        fecha     = get_image_date(base_name)

        with rasterio.open(f_path) as src:
            img  = src.read().astype("float32")
            h, w = img.shape[1], img.shape[2]
            meta = src.meta

        mask_2d   = load_mask(f_path, h, w)
        valid_idx = np.where(mask_2d.flatten())[0]
        X_all     = build_feature_stack(img, cfg)
        X_valid   = X_all[valid_idx]

        if len(X_valid) == 0:
            logger.warning(f"Sin píxeles válidos para inferencia: {base_name}")
            continue

        X_spec_valid = np.nan_to_num(X_valid[:, :cfg.N_SPECTRAL_FEATURES], nan=0.0, posinf=0.0, neginf=0.0)
        X_spec_scaled = scaler.transform(X_spec_valid)
        X_scaled = np.hstack([X_spec_scaled, X_valid[:, cfg.N_SPECTRAL_FEATURES:]])
        X_final  = add_temporal_features(X_scaled, mes, anio).astype(np.float32)

        m_vec = get_meteo_vector(fecha, meteo_df, meteo_scaler) if fecha and meteo_df is not None else None

        with torch.no_grad():
            X_tensor     = torch.from_numpy(X_final).to(device)
            M_tensor     = build_meteo_tensor(len(X_valid), m_vec, device)
            recon        = model(X_tensor, M_tensor)
            scores_valid = compute_weighted_anomaly_score(X_tensor, recon, cfg).cpu().numpy()

        scores_valid = normalize_scores(scores_valid)

        score_map            = np.zeros(h * w, dtype=np.float32)
        score_map[valid_idx] = scores_valid
        score_map            = score_map.reshape(h, w)

        pcts = compute_percentiles(scores_valid)

        # Métricas derivadas
        contrast_score = round(pcts["p99"] - pcts["p50"], 6)
        spread         = round(pcts["p99"] - pcts["p90"], 6)
        anomaly_ratio  = round(pcts["p90"] / (float(np.median(scores_valid)) + 1e-8), 6)

        # Medianas por feature espectral
        per_feature_medians: Dict[str, float] = {}
        for fi, fname in enumerate(cfg.FEATURE_NAMES):
            if fi < X_valid.shape[1]:
                per_feature_medians[f"median_{fname}"] = round(float(np.median(X_valid[:, fi])), 6)

        if save_images:
            save_outputs(score_map, mask_2d, meta, base_name, "AE", exp_folder)
            save_outputs_tif(score_map, mask_2d, meta, base_name, "AE", exp_folder)
            save_outputs_with_parcelas(
                score_map, mask_2d, meta, base_name, "AE", exp_folder, GEOJSON_FEATURES
            )
            #save_percentile_plot(scores_valid, base_name, "AE", exp_folder, pcts=pcts)
            #save_boxplot(scores_valid, base_name, "AE", exp_folder, pcts=pcts)
            save_percentile_heatmaps(
                scores_valid, valid_idx, h, w, mask_2d, meta, base_name, "AE", exp_folder
            )

        logger.info(
            f"   ✅ AE | P50: {pcts['p50']:.4f} | P90: {pcts['p90']:.4f} | "
            f"P95: {pcts['p95']:.4f} | P99: {pcts['p99']:.4f} | "
            f"contrast: {contrast_score:.4f}"
        )

        row = {
            "experimento":    exp_label,
            "bandas_extra":   "|".join(extra_bands) if extra_bands else "RGB_only",
            "indices_activos": "|".join(indices)    if indices     else "none",
            "imagen":         base_name,
            "mes":            str(mes).zfill(2),
            "anio":           str(anio),
            "algoritmo":      "AE",
            "p50":            round(pcts["p50"], 6),
            "p90":            round(pcts["p90"], 6),
            "p95":            round(pcts["p95"], 6),
            "p99":            round(pcts["p99"], 6),
            "contrast_score": contrast_score,
            "spread":         spread,
            "anomaly_ratio":  anomaly_ratio,
        }
        row.update(per_feature_medians)
        summary_rows.append(row)

        # ── Métricas por parcela individual ────
        p_rows = compute_parcela_metrics(
            score_map        = score_map,
            valid_mask       = mask_2d,
            meta             = meta,
            geojson_features = GEOJSON_FEATURES,
            base_name        = base_name,
            exp_label        = exp_label,
            extra_bands      = extra_bands,
            indices          = indices,
            mes              = mes,
            anio             = anio,
        )
        parcela_rows.extend(p_rows)
        logger.info(f"   🗺️  Métricas por parcela: {len(p_rows)} parcelas procesadas.")

    if summary_rows and save_images:
        os.makedirs(exp_folder, exist_ok=True)
        summary_path = os.path.join(exp_folder, "AE_percentiles_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info(f"📋 CSV resumen global guardado: {summary_path}")

    if parcela_rows and save_images:
        os.makedirs(exp_folder, exist_ok=True)
        parcela_path = os.path.join(exp_folder, "AE_parcelas_metrics.csv")
        pd.DataFrame(parcela_rows).to_csv(parcela_path, index=False)
        logger.info(f"📋 CSV métricas por parcela guardado: {parcela_path}")

    logger.info("🏁 Pipeline completado.")
    return summary_rows, parcela_rows


# ----
# ABLATION STUDY
# ----

def _make_ablation_cfg(base_cfg: PipelineConfig, extra_bands: List[str], indices: List[str]) -> PipelineConfig:
    """Crea un PipelineConfig con bandas e índices activos específicos."""
    return PipelineConfig(
        use_extra_bands      = len(extra_bands) > 0,
        use_indices          = len(indices) > 0,
        use_meteo            = base_cfg.use_meteo,
        results_dir          = base_cfg.results_dir,
        max_per_image        = base_cfg.max_per_image,
        val_split            = base_cfg.val_split,
        epochs               = base_cfg.epochs,
        batch_size           = base_cfg.batch_size,
        lr                   = base_cfg.lr,
        weight_decay         = base_cfg.weight_decay,
        patience             = base_cfg.patience,
        min_delta            = base_cfg.min_delta,
        latent_dim           = base_cfg.latent_dim,
        dropout              = base_cfg.dropout,
        grad_clip            = base_cfg.grad_clip,
        scheduler_T0         = base_cfg.scheduler_T0,
        scheduler_Tmult      = base_cfg.scheduler_Tmult,
        scheduler_eta_min    = base_cfg.scheduler_eta_min,
        active_extra_bands   = list(extra_bands),
        active_indices       = list(indices),
        inference_date_start = base_cfg.inference_date_start,
        inference_date_end   = base_cfg.inference_date_end,
    )


def run_ablation_study(
    data_dir:       str,
    meteo_csv_path: Optional[str] = None,
    base_cfg:       Optional[PipelineConfig] = None,
    save_images:    bool = True,
) -> None:
    """
    Ablation study INDIVIDUAL: quita exactamente 1 banda extra a la vez
    (manteniendo todas las demás) y exactamente 1 índice a la vez
    (manteniendo todos los demás). Genera un CSV comparativo final
    con métricas de todas las combinaciones.

    Args:
        data_dir:       Directorio con archivos *_stack.tif.
        meteo_csv_path: Ruta al CSV meteorológico (opcional).
        base_cfg:       Configuración base del pipeline.
        save_images:    Si True, guarda imágenes en la carpeta de cada experimento.
    """
    import traceback
    import datetime

    if base_cfg is None:
        base_cfg = PipelineConfig()

    EXTRA_BANDS_ALL = list(base_cfg.EXTRA_BANDS_ALL)
    ALL_INDICES     = ["NDVI", "GNDVI", "MSAVI2", "NDRE", "NDMI", "EVI", "NDWI", "CRI1"]

    # Construir lista de experimentos
    ablation_configs: List[Tuple[str, List[str], List[str]]] = []
    
    """
    # Experimento solo RGB (sin ninguna banda extra)
    ablation_configs.append(("BANDS_RGB_only", [], list(ALL_INDICES)))
    """
    
    # --- Bandas: RGB + exactamente 1 banda extra a la vez (sin índices) ---
    for band in EXTRA_BANDS_ALL:
        ablation_configs.append((f"RGB_plus_BAND_{band}", [band], []))

    # --- Índices: RGB + exactamente 1 índice a la vez (sin bandas extra) ---
    for idx in ALL_INDICES:
        ablation_configs.append((f"RGB_plus_IDX_{idx}", [], [idx]))
        
    """
    # Configuración base (todo activo)
    ablation_configs.append(("BASE_all_bands_all_indices", list(EXTRA_BANDS_ALL), list(ALL_INDICES)))
    """
    
    """
    # Ablation de bandas extra INDIVIDUAL: quitar exactamente 1 banda, mantener el resto
    for band in EXTRA_BANDS_ALL:
        remaining = [b for b in EXTRA_BANDS_ALL if b != band]
        ablation_configs.append((f"BANDS_remove_{band}", remaining, list(ALL_INDICES)))
    """

        
    # Ablation de índices INDIVIDUAL: quitar exactamente 1 índice, mantener el resto
    #for idx in ALL_INDICES:
    #    remaining = [i for i in ALL_INDICES if i != idx]
    #    ablation_configs.append((f"IDX_remove_{idx}", list(EXTRA_BANDS_ALL), remaining))

    """
    # Experimento sin ningún índice
    ablation_configs.append(("IDX_none", list(EXTRA_BANDS_ALL), []))
    """
    
    """
    # Ablation de bandas extra: quitar 1 a la vez hasta solo RGB
    for i in range(len(EXTRA_BANDS_ALL), 0, -1):
        remaining = EXTRA_BANDS_ALL[:i - 1]
        removed   = EXTRA_BANDS_ALL[i - 1]
        ablation_configs.append((f"BANDS_remove_{str(removed)}", list(remaining), list(ALL_INDICES)))
    
    # Ablation de índices: quitar 1 a la vez hasta ninguno
    for i in range(len(ALL_INDICES), 0, -1):
        remaining = ALL_INDICES[:i - 1]
        removed   = ALL_INDICES[i - 1]
        ablation_configs.append((f"IDX_remove_{str(removed)}", list(EXTRA_BANDS_ALL), list(remaining)))
    """ 

    logger.info(f"🔬 Ablation study INDIVIDUAL: {len(ablation_configs)} configuraciones a evaluar.")

    all_ablation_rows: List[Dict] = []
    all_parcela_rows: List[Dict] = []
    error_log: List[str] = []

    for exp_label, extra_bands, indices in ablation_configs:
        logger.info(
            f"\n{'='*60}\n"
            f"🔬 Experimento: {exp_label}\n"
            f"   Bandas extra : {extra_bands if extra_bands else '(ninguna - solo RGB)'}\n"
            f"   Índices      : {indices if indices else '(ninguno)'}\n"
            f"{'='*60}"
        )
        exp_cfg = _make_ablation_cfg(base_cfg, extra_bands, indices)

        try:
            rows, p_rows = run_anomaly_pipeline(
                data_dir       = data_dir,
                meteo_csv_path = meteo_csv_path,
                cfg            = exp_cfg,
                save_images    = save_images,
                exp_label      = exp_label,
            )
            if not rows:
                msg = f"[{exp_label}] Pipeline completado pero sin filas de resultado."
                logger.warning(f"⚠️  {msg}")
                error_log.append(f"WARNING | {datetime.datetime.now()} | {msg}")
            else:
                all_ablation_rows.extend(rows)
                all_parcela_rows.extend(p_rows)
        except Exception as e:
            tb = traceback.format_exc()
            msg = (
                f"ERROR | {datetime.datetime.now()} | Experimento: '{exp_label}'\n"
                f"  Bandas extra : {extra_bands}\n"
                f"  Índices      : {indices}\n"
                f"  Excepción    : {type(e).__name__}: {e}\n"
                f"  Traceback:\n{tb}"
            )
            logger.error(f"❌ Error en experimento '{exp_label}': {e}")
            error_log.append(msg)
            continue

    # ── Guardar log de errores si los hay ────
    if error_log:
        os.makedirs(base_cfg.results_dir, exist_ok=True)
        error_log_path = os.path.join(base_cfg.results_dir, "ablation_errors.txt")
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Ablation Study - Log de Errores ===\n")
            f.write(f"Generado: {datetime.datetime.now()}\n")
            f.write(f"Total errores/warnings: {len(error_log)}\n")
            f.write("=" * 60 + "\n\n")
            for entry in error_log:
                f.write(entry + "\n" + "-" * 60 + "\n")
        logger.warning(f"⚠️  {len(error_log)} error(es)/warning(s) guardados en: {error_log_path}")
    else:
        logger.info("✅ Todos los experimentos completados sin errores.")

    # ── CSV comparativo final ────
    if all_ablation_rows:
        os.makedirs(base_cfg.results_dir, exist_ok=True)
        df = pd.DataFrame(all_ablation_rows)

        # Columnas de métricas derivadas (ya calculadas en run_anomaly_pipeline)
        # Agregar ranking compuesto por experimento
        ranking = df.groupby(["experimento", "bandas_extra", "indices_activos"]).agg(
            mean_contrast   = ("contrast_score", "mean"),
            mean_spread     = ("spread",         "mean"),
            mean_p99        = ("p99",             "mean"),
            mean_p50        = ("p50",             "mean"),
            mean_ratio      = ("anomaly_ratio",   "mean"),
            n_imagenes      = ("imagen",          "count"),
        ).reset_index()

        for col in ["mean_contrast", "mean_spread", "mean_ratio", "mean_p99"]:
            mn = ranking[col].min()
            mx = ranking[col].max()
            ranking[f"{col}_norm"] = (ranking[col] - mn) / (mx - mn + 1e-8)

        ranking["composite_score"] = ranking[
            ["mean_contrast_norm", "mean_spread_norm", "mean_ratio_norm", "mean_p99_norm"]
        ].mean(axis=1)
        ranking = ranking.sort_values("composite_score", ascending=False).reset_index(drop=True)
        ranking.insert(0, "rank", ranking.index + 1)

        # Guardar detalle completo (todas las imágenes × todos los experimentos)
        front_cols = [
            "experimento", "bandas_extra", "indices_activos", "imagen",
            "mes", "anio", "algoritmo",
            "p50", "p90", "p95", "p99",
            "contrast_score", "spread", "anomaly_ratio",
        ]
        other_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + other_cols]

        detail_path  = os.path.join(base_cfg.results_dir, "ablation_detail_all_images.csv")
        ranking_path = os.path.join(base_cfg.results_dir, "ablation_ranking_experiments.csv")

        df.to_csv(detail_path, index=False)
        ranking.to_csv(ranking_path, index=False)

        # CSV de métricas por parcela (ablation completo)
        if all_parcela_rows:
            parcela_ablation_path = os.path.join(base_cfg.results_dir, "ablation_parcelas_metrics.csv")
            pd.DataFrame(all_parcela_rows).to_csv(parcela_ablation_path, index=False)
            logger.info(f"   🗺️  CSV parcelas ablation: {parcela_ablation_path}")
            
        # ── Ranking por PARCELA ────
        if all_parcela_rows:
            df_parc = pd.DataFrame(all_parcela_rows)

            parcela_ranking = df_parc.groupby(
                ["id_panel", "provincia", "municipio", "poligono", "parcela", "recinto", "dn_surface"]
            ).agg(
                mean_contrast   = ("contrast_score", "mean"),
                mean_spread     = ("spread",         "mean"),
                mean_p99        = ("p99",             "mean"),
                mean_p90        = ("p90",             "mean"),
                mean_p50        = ("p50",             "mean"),
                mean_ratio      = ("anomaly_ratio",   "mean"),
                mean_score      = ("mean_score",      "mean"),
                std_score       = ("std_score",       "mean"),
                n_imagenes      = ("imagen",          "nunique"),
                n_registros     = ("imagen",          "count"),
            ).reset_index()

            # Normalizar y calcular score compuesto
            for col in ["mean_contrast", "mean_spread", "mean_ratio", "mean_p99"]:
                mn = parcela_ranking[col].min()
                mx = parcela_ranking[col].max()
                parcela_ranking[f"{col}_norm"] = (parcela_ranking[col] - mn) / (mx - mn + 1e-8)

            parcela_ranking["composite_score"] = parcela_ranking[
                ["mean_contrast_norm", "mean_spread_norm", "mean_ratio_norm", "mean_p99_norm"]
            ].mean(axis=1)

            parcela_ranking = parcela_ranking.sort_values(
                "composite_score", ascending=False
            ).reset_index(drop=True)
            parcela_ranking.insert(0, "rank", parcela_ranking.index + 1)

            parcela_ranking_path = os.path.join(base_cfg.results_dir, "ablation_ranking_parcelas.csv")
            parcela_ranking.to_csv(parcela_ranking_path, index=False)
            logger.info(f"   🗺️  Ranking por parcela: {parcela_ranking_path}")

            logger.info(f"\n🏆 Top-10 parcelas por composite_score:")
            logger.info(
                parcela_ranking[
                    ["rank", "id_panel", "mean_contrast", "mean_p99", "composite_score"]
                ].head(10).to_string(index=False)
            )    
            
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Ablation study completado.")
        logger.info(f"   📄 Detalle por imagen : {detail_path}")
        logger.info(f"   🏆 Ranking experimentos: {ranking_path}")
        logger.info(f"   Total filas: {len(df)} | Experimentos: {df['experimento'].nunique()}")
        logger.info(f"\n🏆 Top-10 experimentos por composite_score:")
        logger.info(ranking[["rank", "experimento", "mean_contrast", "mean_p99", "composite_score"]].head(10).to_string(index=False))
    else:
        logger.warning("⚠️  Ablation study: no se generaron resultados.")


if __name__ == "__main__":
    DATA_DIR       = "./datasetv3"
    METEO_CSV_PATH = "./Villacarrillo.csv"

    # ── Rango de fechas para INFERENCIA ────
    # El entrenamiento usa TODOS los archivos disponibles.
    # Solo las imágenes dentro de este rango generarán heatmaps y métricas.
    # Poner None para procesar todas las fechas sin restricción.
    INFERENCE_DATE_START = "2024-01-01"   # None → sin límite inferior
    INFERENCE_DATE_END   = "2026-12-31"   # None → sin límite superior

    cfg = PipelineConfig(
        use_extra_bands      = True,
        use_indices          = True,
        use_meteo            = True,
        results_dir          = "results_v7_paper_v2",
        dropout              = 0.5,
        grad_clip            = 1.0,
        scheduler_T0         = 30,
        patience             = 20,
        epochs               = 200,
        inference_date_start = INFERENCE_DATE_START,
        inference_date_end   = INFERENCE_DATE_END,
    )

    # ── Ejecución normal (todas las bandas + todos los índices) ────
    """
    run_anomaly_pipeline(
        data_dir       = DATA_DIR,
        meteo_csv_path = METEO_CSV_PATH,
        cfg            = cfg,
        save_images    = True,
        exp_label      = "BASE_all",
    )
    """

    # ── Ablation study ────
    # Genera ablation_detail_all_images.csv  y  ablation_ranking_experiments.csv
    # en cfg.results_dir con la comparativa de todas las combinaciones.
    # El rango de fechas de inferencia se hereda de cfg.
    run_ablation_study(
        data_dir       = DATA_DIR,
        meteo_csv_path = METEO_CSV_PATH,
        base_cfg       = cfg,
        save_images    = True,   # False para solo métricas, sin imágenes
    )