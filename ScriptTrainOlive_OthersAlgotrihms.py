import numpy as np
import rasterio
import rasterio.transform
import glob
import os
import re
import time
import json
import logging
import traceback
import datetime
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from pyproj import Transformer

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----
# GEOJSON DE PARCELAS (embebido)
# ----
GEOJSON_STR = """{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0454172302099902,38.133864190026486],[-3.04538389621728,38.13389102469108],[-3.0452670281725833,38.13398510401609],[-3.04537473982571,38.13415621471216],[-3.045572701950234,38.13449393586032],[-3.045718525281162,38.13484212935893],[-3.0457698384572467,38.1349646828489],[-3.045819890995019,38.135079395067976],[-3.0459113166093124,38.13529530368643],[-3.0461212631797054,38.13580606085395],[-3.044932121667691,38.13616447692229],[-3.0442996818899912,38.13480731176929],[-3.0442649615304145,38.13475586196816],[-3.0443479105255253,38.13474375598469],[-3.0444294798592915,38.13471560787576],[-3.0444850300876807,38.134681520349915],[-3.044530636863746,38.134623372569564],[-3.044599728057517,38.13452970647242],[-3.044705563002411,38.13425514464484],[-3.0449440021779535,38.13363733380362],[-3.0454172302099902,38.133864190026486]]]},"properties":{"dn_surface":24060.584774877367,"provincia":23,"municipio":48,"agregado":0,"zona":0,"poligono":20,"parcela":112,"recinto":1,"uso_sigpac":"OV","idPanel":"23:48:0:0:20:112:1"},"id":1326710838},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.016743861764303,38.13111071874488],[-3.0173083670509375,38.13121071234113],[-3.0173418611380076,38.13132663574527],[-3.017493817507852,38.13140268978643],[-3.0176060922693364,38.13142169407661],[-3.018397343975892,38.13148920863432],[-3.018520725597931,38.131535624264075],[-3.0185272006185175,38.13156911918935],[-3.0185361198022136,38.131568670757474],[-3.01854344055688,38.13165313100984],[-3.0185620651468374,38.131867997757666],[-3.0195019112416746,38.132087417555596],[-3.0203982393308424,38.1323023312421],[-3.0202160343470297,38.132535531727854],[-3.020156693821799,38.13260216031833],[-3.0198114339894917,38.132989828392255],[-3.019764094673937,38.133040756838966],[-3.0197390428402446,38.133032828396296],[-3.0181555319223126,38.13253167605217],[-3.0177521318533795,38.132369772039056],[-3.0173003355108396,38.13212802201156],[-3.016865354943445,38.13204705282209],[-3.016787753602877,38.13200073777517],[-3.0165218553109083,38.13184205744221],[-3.01655423460479,38.13174782389018],[-3.0166651825109234,38.13142493535878],[-3.0167413538987247,38.13122421474975],[-3.016754125405349,38.13119068629686],[-3.016743861764303,38.13111071874488]]]},"properties":{"dn_surface":30421.815815597707,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":22,"parcela":77,"recinto":4,"uso_sigpac":"OV","idPanel":"23:97:0:0:22:77:4"},"id":1341074656},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0219451548573466,38.119394974914655],[-3.0219422002363028,38.11951776896524],[-3.021951193180751,38.119581563634085],[-3.0219691606294603,38.11965244436428],[-3.022011063442293,38.119746950328164],[-3.0220679472315575,38.119926514174125],[-3.022094893375479,38.12001629693532],[-3.022097912537182,38.12010608304925],[-3.0220873144581795,38.12020858451106],[-3.0193557090487007,38.120265300662254],[-3.0192334204269007,38.11947463820351],[-3.019568580065285,38.11948742898852],[-3.0197226595713285,38.11949330889394],[-3.019837281265841,38.11949768340946],[-3.019930492225617,38.119501240689374],[-3.0200050945210513,38.11950408802207],[-3.020082559236589,38.11950705102501],[-3.020074322339852,38.119428551982566],[-3.0219451548573466,38.119394974914655]]]},"properties":{"dn_surface":21330.68352477426,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":22,"parcela":144,"recinto":1,"uso_sigpac":"OV","idPanel":"23:97:0:0:22:144:1"},"id":1346099366},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0297256049722794,38.149086660837355],[-3.0298643455946057,38.14925813782261],[-3.0300460124602027,38.14945763467375],[-3.030045565704737,38.149481698280994],[-3.0301174832764257,38.14951592997559],[-3.0305930875781524,38.14982702265394],[-3.0309563483867854,38.15004287679005],[-3.0309719881808452,38.15005215639559],[-3.030955846310755,38.150199426443095],[-3.030937883891188,38.15036228850781],[-3.029875353548696,38.149922094396345],[-3.029262186282109,38.14952216611189],[-3.0297256049722794,38.149086660837355]]]},"properties":{"dn_surface":7354.863500327402,"provincia":23,"municipio":48,"agregado":0,"zona":0,"poligono":12,"parcela":59,"recinto":1,"uso_sigpac":"OV","idPanel":"23:48:0:0:12:59:1"},"id":1204495212},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-2.9207468879668053,38.145793073932396],[-2.92083980304538,38.14577430517372],[-2.920962805805374,38.145727992641355],[-2.9211197795827553,38.14563195865642],[-2.9212252683550273,38.14553732361122],[-2.9213550017836685,38.145426441922126],[-2.9214219983397127,38.14536071857728],[-2.9215177465017286,38.14531247317794],[-2.9216201431897364,38.14527714764489],[-2.921734051584247,38.14523848108507],[-2.9219037021527408,38.14518024110379],[-2.9221121198639044,38.14511820076737],[-2.9222175365518055,38.1450905295888],[-2.9224313857365476,38.14504619015656],[-2.92252828976308,38.14505008271263],[-2.9226506638802983,38.145023378805746],[-2.9226991406201805,38.14500140899801],[-2.922801059539679,38.14498161345614],[-2.9229484402283097,38.144926153752294],[-2.9230925016830858,38.14487853531932],[-2.923276427471723,38.14482218713371],[-2.9231900108830637,38.14493257931963],[-2.923087251258626,38.145084630404995],[-2.922912575738881,38.14535704814162],[-2.922791542726251,38.14549526824774],[-2.9227609496160416,38.14553642423297],[-2.922755585197692,38.14556031349662],[-2.9227572624166163,38.1455940934066],[-2.922756370582066,38.145612620766485],[-2.9227192471307064,38.14564657166727],[-2.922648484585344,38.14576161497155],[-2.922376375302754,38.145931114665785],[-2.9223682406652416,38.145936181526565],[-2.9223547022144314,38.14594712074904],[-2.922262357943496,38.14602173058819],[-2.922057033154788,38.146144072854185],[-2.9218959480536553,38.146196598053535],[-2.92180435983043,38.14614588418354],[-2.921108925017501,38.14642876756514],[-2.920397028145762,38.14609679221192],[-2.92046844028866,38.14601825125999],[-2.920613873861067,38.145981850328695],[-2.9206392492362405,38.14583847702821],[-2.920727709333015,38.14582610030925],[-2.9207468879668053,38.145793073932396]]]},"properties":{"dn_surface":18721.1984537415,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":16,"parcela":178,"recinto":1,"uso_sigpac":"OV","idPanel":"23:97:0:0:16:178:1"},"id":1112590751},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.052895437260445,38.12905047196824],[-3.051843620667227,38.12898097844308],[-3.0518435804340887,38.12892492027132],[-3.0508744940682946,38.12886358569122],[-3.050785556203011,38.12846328105934],[-3.052867817211468,38.12857425662574],[-3.0528781160565086,38.128615079011155],[-3.0528815375495872,38.1287674812983],[-3.0528872925646477,38.128990992303414],[-3.0528930157284746,38.129015053396074],[-3.052895437260445,38.12905047196824]]]},"properties":{"dn_surface":8674.21520294821,"provincia":23,"municipio":48,"agregado":0,"zona":0,"poligono":20,"parcela":270,"recinto":1,"uso_sigpac":"OV","idPanel":"23:48:0:0:20:270:1"},"id":1112039892},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-2.92587740677982,38.1565678174797],[-2.925824767586575,38.15670162116075],[-2.9258101344591454,38.1567249547042],[-2.925797412405754,38.15678343776068],[-2.9258145927937087,38.15694522610352],[-2.9258373345747493,38.15698904669591],[-2.9258801024000407,38.15707143997082],[-2.9258327865538165,38.15702561526527],[-2.925807304728466,38.15698630329885],[-2.9257911259780007,38.15695989946411],[-2.925773942237284,38.156941962190245],[-2.9257525675447322,38.156929891410755],[-2.9257161054255723,38.156906253604234],[-2.9256695841838165,38.15688261663587],[-2.925625745151236,38.156866857818784],[-2.9255902888605165,38.15686057139101],[-2.9254987567960473,38.1568630021431],[-2.9254410884606,38.15687607791281],[-2.9251945733208693,38.15693399770538],[-2.924920314079989,38.156991413745544],[-2.924869518905551,38.15699987946823],[-2.924834733167126,38.15699686198292],[-2.9248020429046173,38.156989653545764],[-2.9247685136135524,38.156974146185775],[-2.924710760620877,38.15694195883748],[-2.9246302071738364,38.15686501128542],[-2.9245478088697974,38.15679108121864],[-2.924468597365534,38.156733914959204],[-2.924312687251728,38.156616145859886],[-2.924189300600547,38.156521846929024],[-2.9238826771532107,38.15626719715175],[-2.9235530872898154,38.15600349575675],[-2.923286699494671,38.15580106691573],[-2.9229900521890855,38.155586484314455],[-2.9228866999640584,38.1555195950468],[-2.9228609658434075,38.155495370507],[-2.9227369086399317,38.155390929472645],[-2.922680244456544,38.155353293886975],[-2.9226204798069886,38.155316998567685],[-2.9225670854042223,38.1552921034756],[-2.9225083265831073,38.15527726750611],[-2.9224259299554487,38.15526075431768],[-2.922352880826764,38.15525142274433],[-2.922322662387649,38.15524801298591],[-2.9222540137583923,38.15523879289191],[-2.9222143656776463,38.15523074626435],[-2.9221857515348932,38.15522193017807],[-2.922403100164419,38.15513549347284],[-2.9226214898263807,38.15519791686225],[-2.9226865057385867,38.15522267030016],[-2.922784642743149,38.15526003515035],[-2.9227984569585894,38.15526527467836],[-2.922929929632242,38.15527157954628],[-2.9230351753293076,38.155272099224305],[-2.9232387876950967,38.15527304805582],[-2.923296456030546,38.155252176277486],[-2.9233597167721217,38.15522400469922],[-2.923597951429215,38.155413881639504],[-2.9236984236321404,38.1553884450764],[-2.9237840757911844,38.155454114777115],[-2.9244695160221763,38.15597360002093],[-2.924540078240041,38.15595057409337],[-2.9247917834545243,38.15614829565723],[-2.9248982135145796,38.15609807464318],[-2.9249517185584737,38.15612821429241],[-2.9249982565640362,38.15615455945385],[-2.9254481669782493,38.156527168599666],[-2.9254890673154033,38.15660001237171],[-2.92587740677982,38.1565678174797]]]},"properties":{"dn_surface":18519.7893078827,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":15,"parcela":8,"recinto":3,"uso_sigpac":"OV","idPanel":"23:97:0:0:15:8:3"},"id":1112589849},{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-3.0154393349663513,38.109241593621164],[-3.0157700513576,38.10939378133156],[-3.015829714586122,38.10942126307913],[-3.0156058859068686,38.10980640233274],[-3.015537702473265,38.1099412403407],[-3.0152268872359023,38.11055611919201],[-3.015725981822668,38.110731630387534],[-3.015481990474134,38.11126097191189],[-3.0154324970093365,38.11132118247875],[-3.015412996510446,38.11134317491758],[-3.015452127427734,38.11136507180274],[-3.0154035073573,38.11149618070178],[-3.0153638693348395,38.111470246253646],[-3.015248978581219,38.111381243009504],[-3.0148910562070346,38.11113455520254],[-3.014765992336928,38.11103782551962],[-3.014718716723842,38.111010163561104],[-3.0146464814780045,38.110982502440805],[-3.0146784902917436,38.10971430365157],[-3.0146848546712017,38.10960164332252],[-3.0147229386887213,38.109533232739324],[-3.014773571254228,38.109470228483985],[-3.0148389249571514,38.10943354005346],[-3.01500054733831,38.10938169881743],[-3.0152388163612076,38.10929506094652],[-3.0154393349663513,38.109241593621164]],[[-3.01518893313789,38.11076260990353],[-3.0150902655588245,38.11086108134603],[-3.015337010362715,38.11094966884775],[-3.015381857738308,38.11089715119212],[-3.015328020771028,38.1108676192309],[-3.0153459538539322,38.11081838644333],[-3.01518893313789,38.11076260990353]]]},"properties":{"dn_surface":15961.23622121131,"provincia":23,"municipio":97,"agregado":0,"zona":0,"poligono":23,"parcela":144,"recinto":1,"uso_sigpac":"OV","idPanel":"23:97:0:0:23:144:1"},"id":1210281789}]}"""

GEOJSON_FEATURES = json.loads(GEOJSON_STR)["features"]

# --- PARÁMETROS GLOBALES ---
RGB_BANDS        = ["B02", "B03", "B04"]
EXTRA_BANDS_ALL  = ["B08", "B05", "B06", "B07", "B11", "B12"]
ALL_INDICES      = ["NDVI", "GNDVI", "MSAVI2", "NDRE", "NDMI", "EVI", "NDWI", "CRI1"]
INVALID_SCL      = [3, 8, 9, 10, 11]
BASE_RESULTS_DIR = "results_others_v7_v2"


# ----
# UTILIDADES DE FECHA Y CARPETAS
# ----

def _extract_date_from_name(base_name: str) -> Optional[pd.Timestamp]:
    """Extrae fecha (YYYY-MM-DD o YYYYMMDD) del nombre de archivo."""
    for part in base_name.split("_"):
        if re.match(r'^\d{4}-\d{2}-\d{2}$', part):
            try:
                return pd.to_datetime(part, format="%Y-%m-%d")
            except Exception:
                pass
        if re.match(r'^\d{8}$', part):
            try:
                return pd.to_datetime(part, format="%Y%m%d")
            except Exception:
                pass
    return None


def _make_experiment_folder(
    base_results_dir: str,
    extra_bands: List[str],
    indices: List[str],
) -> str:
    """Construye y crea la carpeta del experimento con nombre descriptivo."""
    bands_part  = "_".join(RGB_BANDS + extra_bands) if extra_bands else "_".join(RGB_BANDS)
    idx_part    = "_".join(indices) if indices else "none"
    folder_name = f"bands_{bands_part}__idx_{idx_part}"
    if len(folder_name) > 120:
        import hashlib
        short_hash  = hashlib.md5(folder_name.encode()).hexdigest()[:8]
        folder_name = folder_name[:100] + f"__h{short_hash}"
    full_path = os.path.join(base_results_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


# ----
# OVERLAY DE PARCELAS GEOJSON
# ----

def _geo_to_pixel(
    lon: float, lat: float,
    transform: rasterio.transform.Affine,
    img_crs: str,
) -> Tuple[int, int]:
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
    transformer  = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)
    pixel_coords = []
    for lon, lat in ring:
        x, y = transformer.transform(lon, lat)
        col, row = ~transform * (x, y)
        pixel_coords.append((float(col), float(row)))

    img_mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img_mask).polygon(pixel_coords, outline=1, fill=1)
    return np.array(img_mask, dtype=bool)


def compute_percentiles(scores: np.ndarray) -> Dict[str, float]:
    """Calcula percentiles clave (p50, p90, p95, p99)."""
    return {
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


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
    alg_name:         str,
) -> List[Dict]:
    """
    Calcula métricas de anomalía (p50, p90, p95, p99, contrast, spread, ratio, mean, std)
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

        rings = geom["coordinates"]
        if not rings:
            continue

        # Solo el anillo exterior (índice 0) define el área de la parcela
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
            "experimento":      exp_label,
            "bandas_extra":     "|".join(extra_bands) if extra_bands else "RGB_only",
            "indices_activos":  "|".join(indices)     if indices     else "none",
            "imagen":           base_name,
            "mes":              str(mes).zfill(2),
            "anio":             str(anio),
            "algoritmo":        alg_name,
            "id_panel":         id_panel,
            "provincia":        props.get("provincia", ""),
            "municipio":        props.get("municipio", ""),
            "poligono":         props.get("poligono", ""),
            "parcela":          props.get("parcela", ""),
            "recinto":          props.get("recinto", ""),
            "dn_surface":       props.get("dn_surface", ""),
            "n_pixels_validos": int(len(pixel_scores)),
            "p50":              round(pcts["p50"], 6),
            "p90":              round(pcts["p90"], 6),
            "p95":              round(pcts["p95"], 6),
            "p99":              round(pcts["p99"], 6),
            "mean_score":       round(float(np.mean(pixel_scores)), 6),
            "std_score":        round(float(np.std(pixel_scores)),  6),
            "contrast_score":   contrast_score,
            "spread":           spread,
            "anomaly_ratio":    anomaly_ratio,
        })

    return rows_out


def overlay_parcelas_on_image(
    rgb_array: np.ndarray,
    meta: dict,
    geojson_features: List[dict],
    outline_color: Tuple[int, int, int] = (169, 39, 245),
    line_width: int = 2,
    label_color: Tuple[int, int, int] = (255, 255, 255),
    draw_labels: bool = False,
) -> np.ndarray:
    """Dibuja los polígonos del GeoJSON sobre una imagen RGB (H, W, 3) numpy array."""
    img_crs   = str(meta["crs"])
    transform = meta["transform"]
    h, w      = rgb_array.shape[:2]

    pil_img = Image.fromarray(rgb_array, mode="RGB")
    draw    = ImageDraw.Draw(pil_img)
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
            pixel_coords = [
                _geo_to_pixel(lon, lat, transform, img_crs)
                for lon, lat in ring
            ]
            in_bounds = any(0 <= c < w and 0 <= r < h for c, r in pixel_coords)
            if not in_bounds:
                continue
            if len(pixel_coords) >= 2:
                draw.line(pixel_coords + [pixel_coords[0]], fill=outline_color, width=line_width)
            if draw_labels and label and len(pixel_coords) >= 1:
                cx = int(np.mean([p[0] for p in pixel_coords]))
                cy = int(np.mean([p[1] for p in pixel_coords]))
                if 0 <= cx < w and 0 <= cy < h:
                    draw.text((cx + 1, cy + 1), label, fill=(0, 0, 0), font=font)
                    draw.text((cx, cy), label, fill=label_color, font=font)
            parcelas_dibujadas += 1

    logger.info(f"   🗺️  Parcelas superpuestas: {parcelas_dibujadas} de {len(geojson_features)}")
    return np.array(pil_img)


def save_outputs_with_parcelas(
    score_map: np.ndarray,
    mask: np.ndarray,
    meta: dict,
    base_name: str,
    alg_name: str,
    results_dir: str,
    geojson_features: List[dict],
) -> None:
    """Guarda el heatmap de anomalía con los polígonos del GeoJSON superpuestos (PNG)."""
    os.makedirs(results_dir, exist_ok=True)
    rgb_data    = colorize_heatmap_rgb(score_map, mask)
    rgb_data    = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)
    rgb_overlay = overlay_parcelas_on_image(rgb_data, meta, geojson_features)
    path_out    = os.path.join(results_dir, f"{base_name}_{alg_name}_ANOMALY_HEATMAP_PARCELAS.png")
    Image.fromarray(rgb_overlay, mode="RGB").save(path_out)
    logger.info(f"💾 Heatmap con parcelas guardado: {path_out}")


# ----
# CÁLCULO DE ÍNDICES (con bandas activas configurables)
# ----

def calculate_indices_active(
    data: np.ndarray,
    bidx: Dict[str, int],
    active_indices: List[str],
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Calcula solo los índices activos según la configuración de ablation."""
    if not active_indices:
        return {}

    red   = data[bidx["B04"]]
    green = data[bidx["B03"]]
    blue  = data[bidx["B02"]]
    nir   = data[bidx["B08"]] if "B08" in bidx else None
    re    = data[bidx["B05"]] if "B05" in bidx else None
    swir  = data[bidx["B11"]] if "B11" in bidx else None

    all_possible: Dict[str, np.ndarray] = {}
    if nir is not None:
        t     = 2.0 * nir + 1.0
        inner = t ** 2 - 8.0 * (nir - red)
        all_possible.update({
            "NDVI":   (nir - red)   / (nir + red   + eps),
            "GNDVI":  (nir - green) / (nir + green + eps),
            "MSAVI2": (t - np.sqrt(np.maximum(inner, 0) + eps)) / 2.0,
            "EVI":    2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps),
            "NDWI":   (green - nir) / (green + nir + eps),
        })
    if nir is not None and re is not None:
        all_possible.update({
            "NDRE": (nir - re) / (nir + re + eps),
            "CRI1": (1 / (green + eps)) - (1 / (re + eps)),
        })
    if nir is not None and swir is not None:
        all_possible["NDMI"] = (nir - swir) / (nir + swir + eps)

    result = {k: all_possible[k] for k in active_indices if k in all_possible}
    return {k: np.clip(v, -1.0, 1.0) for k, v in result.items()}


# ----
# CLASE AUTOENCODER
# ----

class SeasonalAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),          nn.ReLU(),
            nn.Linear(8, 16),         nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_seasonal_ae(X_scaled, month, year, epochs=50):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_feat     = np.full((X_scaled.shape[0], 1), month / 12.0)
    y_feat     = np.full((X_scaled.shape[0], 1), (year - 2015) / 10.0)
    X_temporal = np.hstack([X_scaled, m_feat, y_feat]).astype(np.float32)
    X_tensor   = torch.from_numpy(X_temporal).to(device)
    model      = SeasonalAutoencoder(X_temporal.shape[1]).to(device)
    optimizer  = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion  = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), X_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
    return scores.cpu().numpy()


# --- NORMALIZACIÓN DE SCORES ---

def normalize_scores(scores, p_lo=1, p_hi=99):
    vmin = np.percentile(scores, p_lo)
    vmax = np.percentile(scores, p_hi)
    if vmax <= vmin:
        return np.zeros_like(scores, dtype=np.float32)
    return np.clip((scores - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


# --- ALGORITMOS ADICIONALES ---

def robust_pca_scores(X):
    pca         = PCA(n_components=0.95)
    X_projected = pca.inverse_transform(pca.fit_transform(X))
    return np.linalg.norm(X - X_projected, axis=1)


def rx_detector(X):
    mu      = np.mean(X, axis=0)
    inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))
    diff    = X - mu
    return np.array([d @ inv_cov @ d.T for d in diff])


# --- FUNCIONES DE VISUALIZACIÓN ---

def colorize_heatmap_rgb(A, valid_mask, p_lo=2, p_hi=98, gamma=1.0):
    H, W = A.shape
    rgb  = np.zeros((3, H, W), dtype=np.uint8)
    mask = valid_mask > 0
    if mask.sum() == 0:
        return rgb

    vals = A[mask]
    vmin = float(np.percentile(vals, p_lo))
    vmax = float(np.percentile(vals, p_hi))
    if vmax <= vmin:
        vmax = vmin + 1e-12

    x = np.clip((A - vmin) / (vmax - vmin), 0.0, 1.0)
    if gamma != 1.0:
        x = x ** (1.0 / gamma)

    r, g, b = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    m1 = x < 0.25
    b[m1] = 128 + (255 - 128) * (x[m1] / 0.25)
    m2 = (x >= 0.25) & (x < 0.5)
    g[m2] = 255 * ((x[m2] - 0.25) / 0.25)
    b[m2] = 255
    m3 = (x >= 0.5) & (x < 0.75)
    r[m3] = 255 * ((x[m3] - 0.5) / 0.25)
    g[m3] = 255
    b[m3] = 255 * (1 - (x[m3] - 0.5) / 0.25)
    m4 = x >= 0.75
    r[m4] = 255
    g[m4] = 255 * (1 - (x[m4] - 0.75) / 0.25)

    for channel in [r, g, b]:
        channel[~mask] = 0
    rgb[0], rgb[1], rgb[2] = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)
    return rgb


def save_scores_csv(scores_valid, base_name, alg_name, folder):
    os.makedirs(folder, exist_ok=True)
    path_csv = os.path.join(folder, f"{base_name}_{alg_name}_scores.csv")
    pd.DataFrame({"anomaly_score": scores_valid}).to_csv(path_csv, index=False)
    logger.info(f"   💾 CSV guardado: {path_csv}")


def save_percentile_plot(scores_valid, base_name, alg_name, folder):
    os.makedirs(folder, exist_ok=True)
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
    path_plot = os.path.join(folder, f"{base_name}_{alg_name}_percentiles.png")
    fig.savefig(path_plot, dpi=150)
    plt.close(fig)
    logger.info(f"   📊 Gráfica de percentiles guardada: {path_plot}")


def save_boxplot(scores_valid, base_name, alg_name, folder):
    os.makedirs(folder, exist_ok=True)
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
    path_box = os.path.join(folder, f"{base_name}_{alg_name}_boxplot.png")
    fig.savefig(path_box, dpi=150)
    plt.close(fig)
    logger.info(f"   📦 Boxplot guardado: {path_box}")


def save_percentile_heatmaps(
    scores_valid: np.ndarray,
    valid_idx: np.ndarray,
    h: int, w: int,
    mask: np.ndarray,
    meta: dict,
    base_name: str,
    alg_name: str,
    folder: str,
) -> None:
    os.makedirs(folder, exist_ok=True)
    for pct, pct_label in [(50, "P50"), (90, "P90"), (95, "P95"), (99, "P99")]:
        threshold   = np.percentile(scores_valid, pct)
        binary_flat = np.zeros(h * w, dtype=np.float32)
        binary_flat[valid_idx] = (scores_valid >= threshold).astype(np.float32)
        binary_map  = binary_flat.reshape(h, w)
        path_rgb    = os.path.join(folder, f"{base_name}_{alg_name}_{pct_label}_HEATMAP_RGB.png")
        rgb_data    = colorize_heatmap_rgb(binary_map, mask)
        rgb_data    = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(rgb_data, mode="RGB").save(path_rgb)
        logger.info(f"🗺️ Heatmap {pct_label} guardado (umbral={threshold:.4f}): {path_rgb}")


def save_outputs(score_map, mask, meta, base_name, alg_name, folder):
    os.makedirs(folder, exist_ok=True)
    path_rgb = os.path.join(folder, f"{base_name}_{alg_name}_ANOMALY_HEATMAP_RGB.png")
    rgb_data = colorize_heatmap_rgb(score_map, mask)
    rgb_data = np.transpose(rgb_data, (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb_data, mode="RGB").save(path_rgb)
    logger.info(f"💾 Guardado: {path_rgb}")


# ----
# PIPELINE PRINCIPAL
# ----

def run_anomaly_pipeline(
    data_dir:           str,
    active_extra_bands: Optional[List[str]] = None,
    active_indices:     Optional[List[str]] = None,
    save_images:        bool = True,
    date_from:          Optional[str] = None,
    date_to:            Optional[str] = None,
    base_results_dir:   str = BASE_RESULTS_DIR,
    exp_label:          str = "BASE",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Pipeline de detección de anomalías con múltiples algoritmos.

    Returns:
        Tupla (summary_rows, parcela_rows):
          - summary_rows: métricas globales por imagen × algoritmo.
          - parcela_rows: métricas por parcela individual por imagen × algoritmo.
    """
    extra_bands    = active_extra_bands if active_extra_bands is not None else EXTRA_BANDS_ALL
    indices_active = active_indices     if active_indices     is not None else ALL_INDICES

    band_order    = RGB_BANDS + extra_bands
    bidx          = {name: i for i, name in enumerate(band_order)}
    feature_names = band_order + indices_active

    exp_folder = _make_experiment_folder(base_results_dir, extra_bands, indices_active)

    ts_from = pd.to_datetime(date_from) if date_from else None
    ts_to   = pd.to_datetime(date_to)   if date_to   else None
    if ts_from or ts_to:
        logger.info(
            f"📅 Rango de fechas: "
            f"{ts_from.date() if ts_from else '—'} → {ts_to.date() if ts_to else '—'}"
        )

    files = sorted(glob.glob(os.path.join(data_dir, "*_stack.tif")))
    if not files:
        logger.error("No se encontraron archivos _stack.tif")
        return [], []

    if ts_from or ts_to:
        filtered = []
        for f in files:
            base = os.path.basename(f).replace("_stack.tif", "")
            ts   = _extract_date_from_name(base)
            if ts is None:
                filtered.append(f)
                continue
            if ts_from and ts < ts_from:
                continue
            if ts_to   and ts > ts_to:
                continue
            filtered.append(f)
        logger.info(f"📂 Archivos tras filtro de fechas: {len(filtered)} / {len(files)}")
        files = filtered

    if not files:
        logger.warning("⚠️  Ningún archivo dentro del rango de fechas especificado.")
        return [], []

    logger.info(
        f"Iniciando proceso para {len(files)} archivos | "
        f"Bandas: {band_order} | Índices: {indices_active} | "
        f"Carpeta: {exp_folder}"
    )

    summary_rows: List[Dict] = []
    parcela_rows: List[Dict] = []

    for f_path in files:
        base_name = os.path.basename(f_path).replace("_stack.tif", "")
        logger.info(f"📂 Cargando: {base_name}")

        with rasterio.open(f_path) as src:
            img  = src.read().astype('float32')
            meta = src.meta
            h, w = img.shape[1], img.shape[2]

        # Máscara SCL + DataMask
        scl_path = f_path.replace("_stack.tif", "_SCL_dataMask.tif")
        mask = np.ones((h, w), dtype=bool)
        if os.path.exists(scl_path):
            with rasterio.open(scl_path) as s_src:
                scl, dm = s_src.read(1), s_src.read(2)
                mask = (dm > 0) & (~np.isin(scl, INVALID_SCL))
        else:
            logger.warning(f"Falta máscara SCL para {base_name}")

        # Construir stack de features con bandas e índices activos
        indices_calc = calculate_indices_active(img, bidx, indices_active)
        stack = (
            [img[bidx[b]] for b in band_order]
            + [indices_calc[i] for i in indices_active if i in indices_calc]
        )
        X_all     = np.stack(stack, axis=0).reshape(len(stack), -1).T
        valid_idx = np.where(mask.flatten())[0]
        X_valid   = X_all[valid_idx]

        if X_valid.shape[0] < 10:
            logger.warning(f"Píxeles insuficientes en {base_name}. Saltando...")
            continue

        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(np.nan_to_num(X_valid))

        try:
            date_ts       = _extract_date_from_name(base_name)
            current_month = date_ts.month if date_ts else 1
            current_year  = date_ts.year  if date_ts else 2026
            mes           = str(current_month).zfill(2)
            anio          = str(current_year)
        except Exception:
            current_month, current_year = 1, 2026
            mes, anio = "01", "2026"

        # --- Ejecución de Algoritmos ---
        algos = {
            "IF":   lambda x: -IsolationForest(contamination=0.05).fit(x).decision_function(x),
            "RX":   rx_detector,
            "LOF":  lambda x: -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(x).decision_function(x),
            "SVM":  lambda x: -OneClassSVM(nu=0.05, kernel="rbf").fit(x).decision_function(x),
            "RPCA": robust_pca_scores,
        }

        for name, func in algos.items():
            logger.info(f"   ⚙️  Calculando {name}...")
            start_alg = time.time()
            try:
                scores_valid = normalize_scores(func(X_scaled))

                score_map            = np.zeros(h * w, dtype=np.float32)
                score_map[valid_idx] = scores_valid
                score_map            = score_map.reshape(h, w)

                pcts           = compute_percentiles(scores_valid)
                contrast_score = round(pcts["p99"] - pcts["p50"], 6)
                spread         = round(pcts["p99"] - pcts["p90"], 6)
                anomaly_ratio  = round(pcts["p90"] / (float(np.median(scores_valid)) + 1e-8), 6)

                # Medianas por feature espectral
                per_feature_medians: Dict[str, float] = {}
                for fi, fname in enumerate(feature_names):
                    if fi < X_valid.shape[1]:
                        per_feature_medians[f"median_{fname}"] = round(float(np.median(X_valid[:, fi])), 6)

                if save_images:
                    save_outputs(score_map, mask, meta, base_name, name, exp_folder)
                    save_percentile_plot(scores_valid, base_name, name, exp_folder)
                    save_boxplot(scores_valid, base_name, name, exp_folder)
                    save_percentile_heatmaps(
                        scores_valid, valid_idx, h, w, mask, meta, base_name, name, exp_folder
                    )
                    save_outputs_with_parcelas(
                        score_map, mask, meta, base_name, name, exp_folder, GEOJSON_FEATURES
                    )

                logger.info(
                    f"   ✅ {name} ({time.time()-start_alg:.1f}s) | "
                    f"P50:{pcts['p50']:.4f} P90:{pcts['p90']:.4f} "
                    f"P95:{pcts['p95']:.4f} P99:{pcts['p99']:.4f} | "
                    f"contrast:{contrast_score:.4f}"
                )

                row = {
                    "experimento":    exp_label,
                    "bandas_extra":   "|".join(extra_bands) if extra_bands else "RGB_only",
                    "indices_activos": "|".join(indices_active) if indices_active else "none",
                    "imagen":         base_name,
                    "mes":            mes,
                    "anio":           anio,
                    "algoritmo":      name,
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
                    valid_mask       = mask,
                    meta             = meta,
                    geojson_features = GEOJSON_FEATURES,
                    base_name        = base_name,
                    exp_label        = exp_label,
                    extra_bands      = extra_bands,
                    indices          = indices_active,
                    mes              = current_month,
                    anio             = current_year,
                    alg_name         = name,
                )
                parcela_rows.extend(p_rows)
                logger.info(f"   🗺️  Métricas por parcela ({name}): {len(p_rows)} parcelas procesadas.")

            except Exception as e:
                logger.error(f"   ❌ Error en {name}: {e}")

    if summary_rows and save_images:
        summary_path = os.path.join(exp_folder, "all_techniques_percentiles_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info(f"📋 CSV resumen global guardado: {summary_path}")

    if parcela_rows and save_images:
        parcela_path = os.path.join(exp_folder, "all_techniques_parcelas_metrics.csv")
        pd.DataFrame(parcela_rows).to_csv(parcela_path, index=False)
        logger.info(f"📋 CSV métricas por parcela guardado: {parcela_path}")

    logger.info("🏁 Proceso completado con éxito.")
    return summary_rows, parcela_rows


# ----
# ABLATION STUDY
# ----

def run_ablation_study(
    data_dir:         str,
    save_images:      bool = True,
    date_from:        Optional[str] = None,
    date_to:          Optional[str] = None,
    base_results_dir: str = BASE_RESULTS_DIR,
) -> None:
    """
    Ablation study INDIVIDUAL: quita exactamente 1 banda extra a la vez
    (manteniendo todas las demás) y exactamente 1 índice a la vez
    (manteniendo todos los demás). Incluye experimentos RGB_only e IDX_none.
    Genera CSVs comparativos finales con métricas globales y por parcela.
    Registra errores en ablation_errors.txt.
    """
    
    # ── Construir lista de experimentos ────
    ablation_configs: List[Tuple[str, List[str], List[str]]] = []

    # --- Bandas: RGB + exactamente 1 banda extra a la vez (sin índices) ---
    for band in EXTRA_BANDS_ALL:
        ablation_configs.append((f"RGB_plus_BAND_{band}", [band], []))

    # --- Índices: RGB + exactamente 1 índice a la vez (sin bandas extra) ---
    for idx in ALL_INDICES:
        ablation_configs.append((f"RGB_plus_IDX_{idx}", [], [idx]))
        
    """    
    # Configuración base (todo activo)
    ablation_configs.append(("BASE_all_bands_all_indices", list(EXTRA_BANDS_ALL), list(ALL_INDICES)))

    # Ablation de bandas extra INDIVIDUAL: quitar exactamente 1 banda, mantener el resto
    for band in EXTRA_BANDS_ALL:
        remaining = [b for b in EXTRA_BANDS_ALL if b != band]
        ablation_configs.append((f"BANDS_remove_{band}", remaining, list(ALL_INDICES)))

    # Experimento solo RGB (sin ninguna banda extra)
    ablation_configs.append(("BANDS_RGB_only", [], list(ALL_INDICES)))

    # Ablation de índices INDIVIDUAL: quitar exactamente 1 índice, mantener el resto
    for idx in ALL_INDICES:
        remaining = [i for i in ALL_INDICES if i != idx]
        ablation_configs.append((f"IDX_remove_{idx}", list(EXTRA_BANDS_ALL), remaining))

    # Experimento sin ningún índice
    ablation_configs.append(("IDX_none", list(EXTRA_BANDS_ALL), []))
    
        
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
    
    logger.info(f"🔬 Ablation study: {len(ablation_configs)} configuraciones a evaluar.")

    all_ablation_rows: List[Dict] = []
    all_parcela_rows:  List[Dict] = []
    errors: List[str] = []

    for exp_label, extra_bands, indices in ablation_configs:
        logger.info(
            f"\n{'='*60}\n"
            f"🔬 Experimento: {exp_label}\n"
            f"   Bandas extra : {extra_bands if extra_bands else '(ninguna - solo RGB)'}\n"
            f"   Índices      : {indices if indices else '(ninguno)'}\n"
            f"{'='*60}"
        )
        try:
            rows, p_rows = run_anomaly_pipeline(
                data_dir           = data_dir,
                active_extra_bands = extra_bands,
                active_indices     = indices,
                save_images        = save_images,
                date_from          = date_from,
                date_to            = date_to,
                base_results_dir   = base_results_dir,
                exp_label          = exp_label,
            )
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"[{datetime.datetime.now().isoformat()}] ❌ Error en '{exp_label}': {e}\n{tb}\n"
            logger.error(msg)
            errors.append(msg)
            continue

        if not rows:
            logger.warning(f"⚠️  Experimento '{exp_label}': sin resultados.")
            continue

        all_ablation_rows.extend(rows)
        all_parcela_rows.extend(p_rows)

    # ── Guardar log de errores ────
    if errors:
        err_path = os.path.join(base_results_dir, "ablation_errors.txt")
        os.makedirs(base_results_dir, exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as f:
            f.writelines(errors)
        logger.warning(f"⚠️  {len(errors)} errores registrados en: {err_path}")

    if all_ablation_rows:
        os.makedirs(base_results_dir, exist_ok=True)
        df = pd.DataFrame(all_ablation_rows)

        # ── Detalle completo (todas las imágenes × todos los experimentos) ────
        front_cols = [
            "experimento", "bandas_extra", "indices_activos", "imagen",
            "mes", "anio", "algoritmo",
            "p50", "p90", "p95", "p99",
            "contrast_score", "spread", "anomaly_ratio",
        ]
        other_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + other_cols]

        detail_path = os.path.join(base_results_dir, "ablation_detail_all_images.csv")
        df.to_csv(detail_path, index=False)

        # ── Ranking por experimento × algoritmo ────
        ranking = df.groupby(["experimento", "bandas_extra", "indices_activos", "algoritmo"]).agg(
            mean_contrast = ("contrast_score", "mean"),
            mean_spread   = ("spread",         "mean"),
            mean_p99      = ("p99",             "mean"),
            mean_p50      = ("p50",             "mean"),
            mean_ratio    = ("anomaly_ratio",   "mean"),
            n_imagenes    = ("imagen",          "count"),
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

        ranking_path = os.path.join(base_results_dir, "ablation_ranking_experiments.csv")
        ranking.to_csv(ranking_path, index=False)

        # ── CSV de métricas por parcela (ablation completo) ────
        if all_parcela_rows:
            parcela_ablation_path = os.path.join(base_results_dir, "ablation_parcelas_metrics.csv")
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

            parcela_ranking_path = os.path.join(BASE_RESULTS_DIR, "ablation_ranking_parcelas.csv")
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
        logger.info(f"   📄 Detalle por imagen  : {detail_path}")
        logger.info(f"   🏆 Ranking experimentos: {ranking_path}")
        logger.info(f"   Total filas: {len(df)} | Experimentos: {df['experimento'].nunique()}")
        logger.info(f"\n🏆 Top-10 por composite_score:")
        logger.info(
            ranking[["rank", "experimento", "algoritmo", "mean_contrast", "mean_p99", "composite_score"]]
            .head(10).to_string(index=False)
        )
    else:
        logger.warning("⚠️  Ablation study: no se generaron resultados.")


if __name__ == "__main__":
    DATA_DIR = "./datasetv3"

    # ── Ejecución normal (todas las bandas + todos los índices) ────
    """
    run_anomaly_pipeline(
        data_dir         = DATA_DIR,
        save_images      = True,
        date_from        = "2024-01-01",
        date_to          = "2026-12-31",
        base_results_dir = BASE_RESULTS_DIR,
        exp_label        = "BASE_all_bands_all_indices",
    )
    """

    # ── Ablation study ────
    run_ablation_study(
        data_dir         = DATA_DIR,
        save_images      = True,
        date_from        = "2024-01-01",
        date_to          = "2026-12-31",
        base_results_dir = BASE_RESULTS_DIR,
    )