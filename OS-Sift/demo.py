#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Esempio di utilizzo della classe OS-SIFT
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os_sift import OSSIFT
import argparse
import os


def parse_arguments():
    """
    Analizza gli argomenti da linea di comando
    """
    parser = argparse.ArgumentParser(description='OS-SIFT: Registrazione di immagini SAR-Ottiche')
    
    parser.add_argument('--optical', type=str, default=None,
                        help='Percorso dell\'immagine ottica')
    parser.add_argument('--sar', type=str, default=None,
                        help='Percorso dell\'immagine SAR')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Parametro di scala base (default: 2.0)')
    parser.add_argument('--ratio', type=float, default=2**(1/3),
                        help='Rapporto di scala tra livelli consecutivi (default: 2^(1/3))')
    parser.add_argument('--m_max', type=int, default=8,
                        help='Numero di livelli di scala (default: 8)')
    parser.add_argument('--d', type=float, default=0.04,
                        help='Parametro della funzione Harris (default: 0.04)')
    parser.add_argument('--thresh_opt', type=float, default=0.00001,
                        help='Soglia di risposta per immagini ottiche (default: 0.00001)')
    parser.add_argument('--thresh_sar', type=float, default=0.00001,
                        help='Soglia di risposta per immagini SAR (default: 0.00001)')
    parser.add_argument('--max_keypoints', type=int, default=5000,
                        help='Numero massimo di keypoint da mantenere (default: 5000)')
    parser.add_argument('--multi_region', action='store_true',
                        help='Usa descrittori multi-regione')
    parser.add_argument('--refine', action='store_true',
                        help='Raffina posizioni dei keypoint')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory di output per le immagini risultanti (default: results)')
    
    return parser.parse_args()


def visualize_keypoints(image, keypoints, title, output_path=None):
    """
    Visualizza i keypoint rilevati
    """
    plt.figure(figsize=(10, 8))
    
    # Mostra immagine in scala di grigi
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    
    # Estrai coordinate keypoint
    kp_x = keypoints[:, 0]
    kp_y = keypoints[:, 1]
    
    # Disegna keypoint
    plt.scatter(kp_x, kp_y, c='r', s=10, marker='o')
    
    plt.title(f'{title} ({len(keypoints)} punti)')
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Immagine salvata: {output_path}")
    
    plt.show()


def main():
    """
    Funzione principale
    """
    # Analizza argomenti
    args = parse_arguments()
    
    # Crea directory di output se non esiste
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Leggi immagini
    optical_path = args.optical
    sar_path = args.sar
    
    try:
            optical_img = cv2.imread(optical_path, cv2.IMREAD_GRAYSCALE)
            sar_img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
            
            if optical_img is None:
                raise ValueError(f"Impossibile leggere l'immagine ottica: {optical_path}")
            if sar_img is None:
                raise ValueError(f"Impossibile leggere l'immagine SAR: {sar_path}")
                
            # Normalizza a [0,1] se necessario
            if optical_img.dtype != np.float32:
                optical_img = optical_img.astype(np.float32) / 255.0
            if sar_img.dtype != np.float32:
                sar_img = sar_img.astype(np.float32) / 255.0
                
    except Exception as e:
            print(f"Errore caricamento immagini: {e}")
            print("Utilizzo immagini di esempio.")
            optical_img = np.random.rand(500, 500).astype(np.float32)
            sar_img = np.random.rand(500, 500).astype(np.float32)
    
    # Crea istanza OS-SIFT con i parametri specificati
    ossift = OSSIFT(
        sigma=args.sigma,
        ratio=args.ratio,
        m_max=args.m_max,
        d=args.d,
        d_sh_optical=args.thresh_opt,
        d_sh_sar=args.thresh_sar,
        max_keypoints=args.max_keypoints,
        is_multi_region=args.multi_region
    )
    
    # Esegui registrazione
    print("Inizio registrazione OS-SIFT...")
    results = ossift.process(optical_img, sar_img, refine_keypoints=args.refine)
    
    # Visualizza risultati
    print("\nRisultati OS-SIFT:")
    print(f"Keypoint ottici: {len(results['keypoints_optical'])}")
    print(f"Keypoint SAR: {len(results['keypoints_sar'])}")
    print(f"Match finali: {len(results['matches_optical'])}")
    print(f"Tempo di esecuzione: {results['execution_time']:.2f} secondi")
    
    # Visualizza keypoint sulle immagini originali
    visualize_keypoints(
        optical_img, 
        results['keypoints_optical'], 
        'Keypoint Immagine Ottica',
        os.path.join(args.output_dir, 'optical_keypoints.png')
    )
    
    visualize_keypoints(
        sar_img, 
        results['keypoints_sar'], 
        'Keypoint Immagine SAR',
        os.path.join(args.output_dir, 'sar_keypoints.png')
    )
    
    # Visualizza match
    keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in results['matches_optical'][:, :2]]
    keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in results['matches_sar'][:, :2]]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints1))]
    
    # Converti immagini in BGR se necessario
    if len(optical_img.shape) == 2:
        optical_img_vis = cv2.cvtColor((optical_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        optical_img_vis = (optical_img * 255).astype(np.uint8)
        
    if len(sar_img.shape) == 2:
        sar_img_vis = cv2.cvtColor((sar_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        sar_img_vis = (sar_img * 255).astype(np.uint8)
    
    matches_img = cv2.drawMatches(
        optical_img_vis, keypoints1, sar_img_vis, keypoints2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Match OS-SIFT ({len(matches)} corrispondenze)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'matches.png'), dpi=300)
    plt.show()
    
    # Salva risultati
    cv2.imwrite(os.path.join(args.output_dir, 'fused.png'), results['fused_image'])
    cv2.imwrite(os.path.join(args.output_dir, 'checkerboard.png'), results['checkerboard'])
    cv2.imwrite(os.path.join(args.output_dir, 'colored.png'), results['colored'])
    
    print(f"Risultati salvati nella directory: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()