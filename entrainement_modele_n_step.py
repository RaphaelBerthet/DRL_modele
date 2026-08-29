from entrainement_reseau_objets_utiles.build_state import build_state
from entrainement_reseau_objets_utiles.entrainement_reseau_neurones_n_step import Reseau_neurones
from entrainement_reseau_objets_utiles.parametres import N_STEP, NB_NEURONES_LAYER1, NB_NEURONES_LAYER2, TAILLE_STATE, NB_ACTIONS_POSSIBLES, NB_PARTIES, p_debut, p_fin
import random
import numpy as np
from collections import deque


def jouer_une_partie(reseau_neurones, p, partie):
    """Joue une partie complète et alimente le réseau en samples."""

    ## initialisation de la partie
    raise NotImplementedError("initialisation pas implementé")
    buffer_local = deque()  # stocke (state1, action, reward) en attente

    while partie_en_cours:
        iteration += 1
        state1 = build_state(TAILLE_STATE)
        action = choisir_action(reseau_neurones, state1, p)
        reward, partie_en_cours = executer_action(action)
        state2 = build_state(TAILLE_STATE)

        buffer_local.append((state1, action, reward, state2, not partie_en_cours))

        # dès qu'on a accumulé n transitions, on peut calculer un sample n-step
        if len(buffer_local) >= N_STEP:
            _emettre_sample_n_step(buffer_local, reseau_neurones, N_STEP)
            buffer_local.popleft()

        reseau_neurones.entrainement_reseau(partie)

    # on enleve les dernieres transitions pour pas avoir d'erreurs de calcul
    '''# à la fin de la partie, vider les transitions restantes (n-step raccourci)
    while buffer_local:
        _emettre_sample_n_step(buffer_local, reseau_neurones, len(buffer_local))
        buffer_local.popleft()'''


def _emettre_sample_n_step(buffer_local, reseau_neurones, n):
    from entrainement_reseau_objets_utiles.parametres import gamma
    state1, action, _, _, _ = buffer_local[0]
    G = 0.0
    for i in range(n):
        _, _, r_i, _, _ = buffer_local[i]
        G += (gamma ** i) * r_i
    # état à n pas plus loin (pour le bootstrap), et si la séquence s'est terminée avant n pas
    _, _, _, state_n, terminal_n = buffer_local[min(n, len(buffer_local)) - 1]
    sample = np.concatenate([state1, state_n, [action, G, terminal_n]])
    reseau_neurones.ajout_sample(sample)

def choisir_action(reseau_neurones, state, p):
    if random.random() <= p:
        return random.randint(1, NB_ACTIONS_POSSIBLES)
    return np.argmax(reseau_neurones.calcul_couche_sortie(state)) + 1


def executer_action(action):
    raise NotImplementedError("executer action pas implementé")


def entrainer(nb_parties=NB_PARTIES):
    reseau_neurones = Reseau_neurones("reseau_neurones.npz", TAILLE_STATE, NB_ACTIONS_POSSIBLES, NB_NEURONES_LAYER1, NB_NEURONES_LAYER2)
    for partie in range(nb_parties):
        p = p_debut - (p_debut - p_fin) * partie / nb_parties
        if partie % 100 == 0:
            print(f"progression : {partie * 100 / nb_parties} %")
        jouer_une_partie(reseau_neurones, p, partie)
    return reseau_neurones


if __name__ == "__main__":
    entrainer()