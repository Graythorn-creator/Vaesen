#!/usr/bin/env python3
"""Prépare le Tarot du Malakara (A4, cartes de 60 x 120 mm) pour l'impression
recto verso.

Les pages recto sont reprises telles quelles. Chaque verso est reconstruit :
- le cadre de l'illustration est recentré sur la carte, avec une bande jaune
  (parchemin) plus large tout autour (--bande-cotes / --bande-haut-bas) ;
- le parchemin est prolongé au-delà du trait de coupe (fond perdu), jusqu'à
  rejoindre la carte voisine, pour qu'un décalage de l'imprimante ne laisse
  pas de liseré blanc ;
- toutes les pages verso peuvent être décalées de (dx, dy) mm pour compenser
  le décalage propre à l'imprimante, mesuré avec la feuille de calibration.

Usage :
    python3 ajuster_recto_verso.py original.pdf sortie.pdf
            [--bande-cotes 2.5] [--bande-haut-bas 3.5] [--fond-perdu 4]
            [--verso-dx 0] [--verso-dy 0]
    python3 ajuster_recto_verso.py --calibration calibration.pdf

Convention du décalage (vu du côté verso) : dx > 0 = vers la droite,
dy > 0 = vers le bas. Ce sont directement les valeurs lues sur la feuille
de calibration (D = +, G = -, B = +, H = -).
"""
import argparse
import io
import re

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

MM = 72 / 25.4
CARTE_L, CARTE_H = 60, 120  # mm
DPI_VERSO = 400


def placements(page):
    """Cartes de la page : (zone de découpe, zone de l'image) en points, repère haut-gauche."""
    h = page.rect.height
    motif = re.compile(
        r"n ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re W\*? n\s*q\s*"
        r"([\d.]+) 0 0 ([\d.]+) ([\d.]+) ([\d.]+) cm\s*/\S+ Do"
    )
    res = []
    for m in motif.finditer(page.read_contents().decode("latin1")):
        cx, cy, cw, ch, a, d, e, f = map(float, m.groups())
        res.append((pymupdf.Rect(cx, h - cy - ch, cx + cw, h - cy), pymupdf.Rect(e, h - f - d, e + a, h - f)))
    return res


def luminance(a):
    return a[..., :3].mean(axis=2)


def rogner_noir(a):
    """Retire les bandes noires (fond derrière la carte) autour de l'illustration."""
    lum = luminance(a)
    lignes = np.nonzero(lum.mean(axis=1) > 60)[0]
    colonnes = np.nonzero(lum.mean(axis=0) > 60)[0]
    return a[lignes[0]:lignes[-1] + 1, colonnes[0]:colonnes[-1] + 1]


def cadre(lum):
    """Bord extérieur du filet sombre qui encadre l'illustration (x0, y0, x1, y1), en pixels."""
    h, w = lum.shape
    sombre = lum < 90
    lignes = range(int(h * 0.2), int(h * 0.8), 7)
    colonnes = range(int(w * 0.2), int(w * 0.8), 7)
    x0 = np.median([np.argmax(sombre[y, : w // 4]) for y in lignes])
    x1 = w - np.median([np.argmax(sombre[y, ::-1][: w // 4]) for y in lignes])
    y0 = np.median([np.argmax(sombre[: h // 8, x]) for x in colonnes])
    y1 = h - np.median([np.argmax(sombre[::-1, x][: h // 8]) for x in colonnes])
    return x0, y0, x1, y1


def ping_pong(i, debut, fin):
    """Ramène les indices i dans [debut, fin) par réflexions successives."""
    n = fin - debut
    k = np.mod(i - debut, 2 * n)
    return debut + np.where(k < n, k, 2 * n - 1 - k)


def nettoyer_bande(bande, axe):
    """Remplace les lignes d'une bande de parchemin touchées par un ornement ou un coin
    par leur reflet côté propre, pour pouvoir la prolonger sans recopier l'ornement."""
    lum = luminance(bande)
    mini = lum.min(axis=1 - axe)
    n = len(mini)
    propre = mini > 95
    propre[:40] = propre[-40:] = False  # coins arrondis et vieillis
    idx = np.arange(n)
    for i in np.nonzero(~propre)[0]:
        s = i
        while s > 0 and not propre[s - 1]:
            s -= 1
        e = i
        while e < n - 1 and not propre[e + 1]:
            e += 1
        for j in (2 * s - 1 - i, 2 * e + 1 - i):
            if 0 <= j < n and propre[j]:
                idx[i] = j
                break
        else:
            idx[i] = np.nonzero(propre)[0][np.argmin(abs(np.nonzero(propre)[0] - i))]
    return np.take(bande, idx, axis=axe)


def verso_elargi(illu, bande_x, bande_y, fond_perdu):
    """Carte verso de 60 x 120 mm + fond perdu tout autour, en image PIL à DPI_VERSO.

    Le cadre de l'illustration est placé à bande_x mm des bords gauche/droit et
    bande_y mm des bords haut/bas ; tout ce qui est au-delà est du parchemin
    prolongé à partir de la bande d'origine."""
    illu = rogner_noir(illu).astype(np.float32)
    h, w = illu.shape[:2]
    fx0, fy0, fx1, fy1 = cadre(luminance(illu))
    sx = (CARTE_L - 2 * bande_x) / (fx1 - fx0)  # mm par pixel source
    sy = (CARTE_H - 2 * bande_y) / (fy1 - fy0)

    # Zone source nécessaire pour couvrir carte + fond perdu (en pixels source).
    bx0 = fx0 + (-fond_perdu - bande_x) / sx
    bx1 = fx0 + (CARTE_L + fond_perdu - bande_x) / sx
    by0 = fy0 + (-fond_perdu - bande_y) / sy
    by1 = fy0 + (CARTE_H + fond_perdu - bande_y) / sy
    marge = int(np.ceil(max(-bx0, bx1 - w, -by0, by1 - h, 0))) + 8

    # Bandes de parchemin d'origine (sans le bord extrême), nettoyées des ornements.
    a, l = 6, 14
    gauche = nettoyer_bande(illu[:, a:a + l], 0)
    droite = nettoyer_bande(illu[:, w - a - l:w - a], 0)
    haut = nettoyer_bande(illu[a:a + l, :], 1)
    bas = nettoyer_bande(illu[h - a - l:h - a, :], 1)

    y, x = np.mgrid[-marge:h + marge, -marge:w + marge]
    yy, xx = ping_pong(y, 0, h), ping_pong(x, 0, w)
    lateral = np.where(
        (x < w / 2)[..., None],
        gauche[yy, ping_pong(x, a, a + l) - a],
        droite[yy, ping_pong(x, w - a - l, w - a) - (w - a - l)],
    )
    vertical = np.where(
        (y < h / 2)[..., None],
        haut[ping_pong(y, a, a + l) - a, xx],
        bas[ping_pong(y, h - a - l, h - a) - (h - a - l), xx],
    )
    ex = np.maximum(np.maximum(a - x, x - (w - 1 - a)), 0)
    ey = np.maximum(np.maximum(a - y, y - (h - 1 - a)), 0)
    poids = np.where(ex + ey > 0, ey / np.maximum(ex + ey, 1), 0.5)[..., None]
    reflet = (1 - poids) * lateral + poids * vertical

    # Au-delà du premier reflet, les reflets successifs forment des motifs
    # répétés : on passe progressivement à un parchemin lissé et légèrement grainé.
    lisse = np.asarray(
        Image.fromarray(np.clip(reflet, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(12))
    ).astype(np.float32)
    grain = np.asarray(
        Image.fromarray(np.clip(128 + np.random.default_rng(0).normal(0, 14, y.shape), 0, 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(0.8))
    ).astype(np.float32)[..., None] - 128
    dehors = np.hypot(np.maximum(np.maximum(-x, x - (w - 1)), 0), np.maximum(np.maximum(-y, y - (h - 1)), 0))
    loin = np.clip((dehors - l / 2) / l, 0, 1)[..., None]
    parchemin = (1 - loin) * reflet + loin * (lisse + grain)

    # L'illustration recouvre le parchemin, avec un fondu sur ses 6 px extérieurs
    # et des coins arrondis (les coins d'origine sont sombres).
    r = 30
    px = np.abs(x - (w - 1) / 2) - (w / 2 - r)
    py = np.abs(y - (h - 1) / 2) - (h / 2 - r)
    interieur = -(np.hypot(np.maximum(px, 0), np.maximum(py, 0)) + np.minimum(np.maximum(px, py), 0) - r)
    alpha = np.clip((interieur - 2) / 4, 0, 1)[..., None]
    source = np.zeros_like(parchemin)
    source[marge:marge + h, marge:marge + w] = illu
    image = alpha * source + (1 - alpha) * parchemin

    ppm = DPI_VERSO / 25.4
    taille = (round((CARTE_L + 2 * fond_perdu) * ppm), round((CARTE_H + 2 * fond_perdu) * ppm))
    boite = (bx0 + marge, by0 + marge, bx1 + marge, by1 + marge)
    tuile = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize(taille, Image.LANCZOS, box=boite)
    etirement = sy / sx
    return tuile, etirement


def voisins(cartes, c):
    """Distance (mm) jusqu'à la carte voisine de chaque côté, None s'il n'y en a pas."""
    g = d = h = b = None
    for o in cartes:
        if o == c:
            continue
        if abs(o.y0 - c.y0) < 1 and o.x1 <= c.x0:
            g = min(g or 1e9, (c.x0 - o.x1) / MM)
        if abs(o.y0 - c.y0) < 1 and o.x0 >= c.x1:
            d = min(d or 1e9, (o.x0 - c.x1) / MM)
        if abs(o.x0 - c.x0) < 1 and o.y1 <= c.y0:
            h = min(h or 1e9, (c.y0 - o.y1) / MM)
        if abs(o.x0 - c.x0) < 1 and o.y0 >= c.y1:
            b = min(b or 1e9, (o.y0 - c.y1) / MM)
    return g, h, d, b


def ajuster(entree, sortie, bande_x, bande_y, fond_perdu, verso_dx_mm, verso_dy_mm):
    src = pymupdf.open(entree)

    # Illustration du verso (la même sur toutes les pages verso).
    xref = src[1].get_images(full=True)[0][0]
    pix = pymupdf.Pixmap(src, xref)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    illu = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
    tuile, etirement = verso_elargi(illu, bande_x, bande_y, fond_perdu)
    print(f"verso : bande {bande_x} mm (côtés) / {bande_y} mm (haut, bas), étirement vertical {100 * (etirement - 1):+.1f} %")

    # Page intermédiaire contenant une carte verso et son fond perdu : chaque
    # carte en affiche une découpe, l'image n'est donc incluse qu'une fois.
    png = io.BytesIO()
    tuile.save(png, "PNG", optimize=True)
    tuile_doc = pymupdf.open()
    tuile_page = tuile_doc.new_page(width=(CARTE_L + 2 * fond_perdu) * MM, height=(CARTE_H + 2 * fond_perdu) * MM)
    tuile_page.insert_image(tuile_page.rect, stream=png.getvalue())

    final = pymupdf.open()
    for pno in range(len(src)):
        if pno % 2 == 0:
            final.insert_pdf(src, from_page=pno, to_page=pno)
            continue
        largeur = src[pno].rect.width
        # Emplacement exact des cartes : miroir des zones de découpe du recto.
        cartes = [pymupdf.Rect(largeur - r.x1, r.y0, largeur - r.x0, r.y1) for r, _ in placements(src[pno - 1])]
        page = final.new_page(width=largeur, height=src[pno].rect.height)
        for c in cartes:
            fp = [fond_perdu if v is None else min(fond_perdu, v / 2) for v in voisins(cartes, c)]
            clip = pymupdf.Rect(
                (fond_perdu - fp[0]) * MM, (fond_perdu - fp[1]) * MM,
                (fond_perdu + CARTE_L + fp[2]) * MM, (fond_perdu + CARTE_H + fp[3]) * MM,
            )
            cible = pymupdf.Rect(c.x0 - fp[0] * MM, c.y0 - fp[1] * MM, c.x1 + fp[2] * MM, c.y1 + fp[3] * MM)
            cible += (verso_dx_mm * MM, verso_dy_mm * MM, verso_dx_mm * MM, verso_dy_mm * MM)
            page.show_pdf_page(cible, tuile_doc, 0, clip=clip)
    final.set_metadata(src.metadata)
    final.save(sortie, garbage=4, deflate=True)


def calibration(sortie):
    """Feuille de test : croix au recto, règles graduées au verso aux mêmes endroits."""
    doc = pymupdf.open()
    largeur, hauteur = pymupdf.paper_size("a4")
    cibles = {
        "1": (40 * MM, 60 * MM), "2": (largeur - 40 * MM, 60 * MM),
        "3": (largeur / 2, hauteur / 2),
        "4": (40 * MM, hauteur - 45 * MM), "5": (largeur - 40 * MM, hauteur - 45 * MM),
    }
    noir = (0, 0, 0)

    recto = doc.new_page(width=largeur, height=hauteur)
    texte = (
        "TEST D'ALIGNEMENT RECTO VERSO - page 1 (recto)\n\n"
        "1. Imprimez ces 2 pages en recto verso, retournement sur les BORDS LONGS,\n"
        "    à 100 % / taille réelle, avec le même papier et les mêmes réglages que les cartes.\n"
        "2. Regardez la page 2 (verso) par transparence, contre une fenêtre ou une lampe.\n"
        "3. Pour chaque cible, lisez où la croix du recto tombe sur les règles du verso."
    )
    recto.insert_textbox(pymupdf.Rect(20 * MM, 90 * MM, largeur - 20 * MM, 140 * MM), texte, fontsize=9, fontname="helv")
    for nom, (x, y) in cibles.items():
        recto.draw_line((x - 12 * MM, y), (x + 12 * MM, y), color=noir, width=0.3)
        recto.draw_line((x, y - 12 * MM), (x, y + 12 * MM), color=noir, width=0.3)
        recto.draw_circle((x, y), 1 * MM, color=noir, width=0.3)
        recto.insert_text((x + 2 * MM, y - 2 * MM), nom, fontsize=9, fontname="hebo")

    verso = doc.new_page(width=largeur, height=hauteur)
    texte = (
        "TEST D'ALIGNEMENT RECTO VERSO - page 2 (verso)\n\n"
        "Par transparence, notez pour chaque cible (1 à 5) où tombe la croix du recto :\n"
        "sur la règle horizontale : G (gauche) ou D (droite) + nombre de mm,\n"
        "sur la règle verticale : H (haut) ou B (bas) + nombre de mm.\n"
        "Graduations tous les mm. Exemple : « 3 : D 2, B 1 ». Croix pile au centre = aligné."
    )
    verso.insert_textbox(pymupdf.Rect(20 * MM, 90 * MM, largeur - 20 * MM, 140 * MM), texte, fontsize=9, fontname="helv")
    for nom, (x, y) in cibles.items():
        x = largeur - x  # le verso est retourné sur le bord long : miroir horizontal
        verso.draw_line((x - 10 * MM, y), (x + 10 * MM, y), color=noir, width=0.25)
        verso.draw_line((x, y - 10 * MM), (x, y + 10 * MM), color=noir, width=0.25)
        for i in range(-10, 11):
            t = (1.6 if i % 5 == 0 else 0.9) * MM
            verso.draw_line((x + i * MM, y - t), (x + i * MM, y + t), color=noir, width=0.2)
            verso.draw_line((x - t, y + i * MM), (x + t, y + i * MM), color=noir, width=0.2)
            if i and i % 5 == 0:
                n = str(abs(i))
                verso.insert_text((x + i * MM - 1.3 * len(n), y - 2.3 * MM), n, fontsize=5, fontname="helv")
                verso.insert_text((x + 2.3 * MM, y + i * MM + 1.7), n, fontsize=5, fontname="helv")
        verso.insert_text((x - 13 * MM, y + 1.2 * MM), "G", fontsize=8, fontname="hebo")
        verso.insert_text((x + 11 * MM, y + 1.2 * MM), "D", fontsize=8, fontname="hebo")
        verso.insert_text((x - 1 * MM, y - 11 * MM), "H", fontsize=8, fontname="hebo")
        verso.insert_text((x - 1 * MM, y + 13.5 * MM), "B", fontsize=8, fontname="hebo")
        verso.insert_text((x - 7 * MM, y - 6 * MM), nom, fontsize=9, fontname="hebo")
    doc.save(sortie, garbage=4, deflate=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entree", nargs="?")
    p.add_argument("sortie", nargs="?")
    p.add_argument("--bande-cotes", type=float, default=2.5, help="bande jaune du verso à gauche et à droite, en mm (défaut 2.5)")
    p.add_argument("--bande-haut-bas", type=float, default=3.5, help="bande jaune du verso en haut et en bas, en mm (défaut 3.5)")
    p.add_argument("--fond-perdu", type=float, default=4, help="fond perdu du verso vers l'extérieur de la feuille, en mm (défaut 4)")
    p.add_argument("--verso-dx", type=float, default=0, help="décalage horizontal du verso en mm (+ = droite)")
    p.add_argument("--verso-dy", type=float, default=0, help="décalage vertical du verso en mm (+ = bas)")
    p.add_argument("--calibration", metavar="PDF", help="génère la feuille de test d'alignement")
    a = p.parse_args()
    if a.calibration:
        calibration(a.calibration)
    if a.entree and a.sortie:
        ajuster(a.entree, a.sortie, a.bande_cotes, a.bande_haut_bas, a.fond_perdu, a.verso_dx, a.verso_dy)
