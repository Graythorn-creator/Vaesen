#!/usr/bin/env python3
"""Prépare le Tarot du Malakara (A4, 3 x 2 cartes de 60 x 120 mm) pour
l'impression recto verso.

- Ajoute un fond perdu autour des cartes (beige au recto, sombre au verso)
  pour qu'un léger décalage de l'imprimante ne laisse pas de liseré blanc.
- Replace les traits de coupe à l'extérieur de ce fond perdu.
- Peut décaler toutes les pages verso de (dx, dy) mm pour compenser le
  décalage propre à l'imprimante (mesuré avec la feuille de calibration).

Usage :
    python3 ajuster_recto_verso.py original.pdf sortie.pdf [--fond-perdu 5]
            [--verso-dx 0] [--verso-dy 0]
    python3 ajuster_recto_verso.py --calibration calibration.pdf

Convention du décalage (vu du côté verso) : dx > 0 = vers la droite,
dy > 0 = vers le bas. Ce sont directement les valeurs lues sur la feuille
de calibration (D = +, G = -, B = +, H = -).
"""
import argparse

import pymupdf

MM = 72 / 25.4
CARTE_L, CARTE_H = 60 * MM, 120 * MM
GRIS_COUPE = (0.533333, 0.533333, 0.533333)


def cellules(page):
    """Cartes de la page : rectangles pleins de la taille d'une carte."""
    res = []
    for d in page.get_drawings():
        r = d["rect"]
        if d["type"] == "f" and abs(r.width - CARTE_L) < 1 and abs(r.height - CARTE_H) < 1:
            res.append((r, d["fill"]))
    return res


def traits_de_coupe(page, grille, xs, ys, marge):
    """Petits traits gris dans la marge, dans le prolongement des lignes de coupe."""
    debut, fin = marge + 1 * MM, marge + 4 * MM
    for x in xs:
        page.draw_line((x, grille.y0 - debut), (x, grille.y0 - fin), color=GRIS_COUPE, width=0.35)
        page.draw_line((x, grille.y1 + debut), (x, grille.y1 + fin), color=GRIS_COUPE, width=0.35)
    for y in ys:
        page.draw_line((grille.x0 - debut, y), (grille.x0 - fin, y), color=GRIS_COUPE, width=0.35)
        page.draw_line((grille.x1 + debut, y), (grille.x1 + fin, y), color=GRIS_COUPE, width=0.35)


def ajuster(entree, sortie, fond_perdu_mm, verso_dx_mm, verso_dy_mm):
    src = pymupdf.open(entree)
    fond_perdu = fond_perdu_mm * MM

    # La grille (identique sur toutes les pages) : union des cartes de la 1re page.
    toutes = cellules(src[0])
    grille = pymupdf.Rect(toutes[0][0])
    for r, _ in toutes[1:]:
        grille |= r
    xs = sorted({round(r.x0, 2) for r, _ in toutes} | {round(r.x1, 2) for r, _ in toutes})
    ys = sorted({round(r.y0, 2) for r, _ in toutes} | {round(r.y1, 2) for r, _ in toutes})

    # 1) Chaque page reconstruite sans décalage : fond perdu, page d'origine
    #    découpée à la grille (on retire ainsi les anciens traits de coupe),
    #    puis nouveaux traits de coupe.
    propre = pymupdf.open()
    for pno, page in enumerate(src):
        neuve = propre.new_page(width=page.rect.width, height=page.rect.height)
        for r, couleur in cellules(page):
            elargi = pymupdf.Rect(r.x0 - fond_perdu, r.y0 - fond_perdu, r.x1 + fond_perdu, r.y1 + fond_perdu)
            neuve.draw_rect(elargi, color=None, fill=couleur, width=0)
        neuve.show_pdf_page(grille, src, pno, clip=grille)
        traits_de_coupe(neuve, grille, xs, ys, fond_perdu)

    # 2) Assemblage final : les pages impaires (verso) sont décalées si demandé.
    final = pymupdf.open()
    for pno, page in enumerate(propre):
        neuve = final.new_page(width=page.rect.width, height=page.rect.height)
        cible = pymupdf.Rect(page.rect)
        if pno % 2 == 1:
            cible += (verso_dx_mm * MM, verso_dy_mm * MM, verso_dx_mm * MM, verso_dy_mm * MM)
        neuve.show_pdf_page(cible, propre, pno)
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
    p.add_argument("--fond-perdu", type=float, default=5, help="fond perdu en mm (défaut 5)")
    p.add_argument("--verso-dx", type=float, default=0, help="décalage horizontal du verso en mm (+ = droite)")
    p.add_argument("--verso-dy", type=float, default=0, help="décalage vertical du verso en mm (+ = bas)")
    p.add_argument("--calibration", metavar="PDF", help="génère la feuille de test d'alignement")
    a = p.parse_args()
    if a.calibration:
        calibration(a.calibration)
    if a.entree and a.sortie:
        ajuster(a.entree, a.sortie, a.fond_perdu, a.verso_dx, a.verso_dy)
