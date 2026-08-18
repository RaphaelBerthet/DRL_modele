from entrainement_reseau_objets_utiles.build_state import build_state
from entrainement_reseau_objets_utiles.entrainement_reseau_neurones import Reseau_neurones
from entrainement_reseau_objets_utiles.parametres import TAILLE_STATE, NB_ACTIONS_POSSIBLES, NB_PARTIES, p_debut, p_fin
import random
import numpy as np


def jouer_une_partie(reseau_neurones, p, partie):
    """Joue une partie complète et alimente le réseau en samples."""
    ## initialisation de la partie
    raise NotImplementedError("initialisation pas implementé")

    partie_en_cours = True
    while partie_en_cours:
        state1 = build_state(TAILLE_STATE)
        action = choisir_action(reseau_neurones, state1, p)
        reward, partie_en_cours = executer_action(action)
        state2 = build_state(TAILLE_STATE)
        sample = np.concatenate([state1, state2, [action, reward, not partie_en_cours]])
        reseau_neurones.ajout_sample(sample)
        reseau_neurones.entrainement_reseau(partie)


def choisir_action(reseau_neurones, state, p):
    if random.random() <= p:
        return random.randint(1, NB_ACTIONS_POSSIBLES)
    return np.argmax(reseau_neurones.calcul_couche_sortie(state)) + 1


def executer_action(action):
    raise NotImplementedError("executer action pas implementé")


def entrainer(nb_parties=NB_PARTIES):
    reseau_neurones = Reseau_neurones()
    for partie in range(nb_parties):
        p = p_debut - (p_debut - p_fin) * partie / nb_parties
        if partie % 100 == 0:
            print(f"progression : {partie * 100 / nb_parties} %")
        jouer_une_partie(reseau_neurones, p, partie)
    return reseau_neurones


if __name__ == "__main__":
    entrainer()