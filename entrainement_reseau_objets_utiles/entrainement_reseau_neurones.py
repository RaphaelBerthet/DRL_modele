import numpy as np
from .adam_update import adam_update
from .relu import relu, relu_derivative
from .parametres import NB_SAMPLES_MAX, NB_SAMPLES_DEBUT_ENTRAINEMENT, TAILLE_BATCHS, NB_ENTRAINEMENT_BATCH, gamma, learning_rate, ACTU_W_TARGET, PERIODE_STOCKAGE_PC, MAX_NORME_GRADIENT, DELTA_HUBER_LOSS
from numpy.typing import NDArray


class Reseau_neurones:
    def __init__(self, nom_fichier: str, TAILLE_STATE: int, NB_ACTIONS_POSSIBLES: int, NB_NEURONES_LAYER1: int, NB_NEURONES_LAYER2: int):
        self.nom_fichier = nom_fichier
        self.NB_ACTIONS_POSSIBLES = NB_ACTIONS_POSSIBLES
        self.NB_NEURONES_LAYER1 = NB_NEURONES_LAYER1
        self.NB_NEURONES_LAYER2 = NB_NEURONES_LAYER2
        self.TAILLE_STATE = TAILLE_STATE
        self.TAILLE_SAMPLE = 2 * TAILLE_STATE + 3

        try:
            data = np.load(nom_fichier)
            self.w1 = data['W1']
            self.w2 = data['W2']
            self.w3 = data['W3']
            self.b1 = data['B1']
            self.b2 = data['B2']
            self.b3 = data['B3']
            self.mW1 = data['mW1']; self.vW1 = data['vW1']
            self.mW2 = data['mW2']; self.vW2 = data['vW2']
            self.mW3 = data['mW3']; self.vW3 = data['vW3']
            self.mB1 = data['mB1']; self.vB1 = data['vB1']
            self.mB2 = data['mB2']; self.vB2 = data['vB2']
            self.mB3 = data['mB3']; self.vB3 = data['vB3']
            self.t_adam = int(data['t_adam'])

        except FileNotFoundError:
            self.w1 = (np.random.randn(self.NB_NEURONES_LAYER1, self.TAILLE_STATE) * np.sqrt(2 / self.TAILLE_STATE)).astype(np.float32)  # He init pour ReLU
            self.w2 = (np.random.randn(self.NB_NEURONES_LAYER2, self.NB_NEURONES_LAYER1) * np.sqrt(2 / self.NB_NEURONES_LAYER1)).astype(np.float32)
            self.w3 = (np.random.randn(self.NB_ACTIONS_POSSIBLES, self.NB_NEURONES_LAYER2) * np.sqrt(2 / self.NB_NEURONES_LAYER2)).astype(np.float32)
            self.b1 = np.zeros(self.NB_NEURONES_LAYER1, dtype=np.float32)
            self.b2 = np.zeros(self.NB_NEURONES_LAYER2, dtype=np.float32)
            self.b3 = np.zeros(self.NB_ACTIONS_POSSIBLES, dtype=np.float32)
            self.mW1 = np.zeros_like(self.w1, dtype=np.float32); self.vW1 = np.zeros_like(self.w1, dtype=np.float32)
            self.mW2 = np.zeros_like(self.w2, dtype=np.float32); self.vW2 = np.zeros_like(self.w2, dtype=np.float32)
            self.mW3 = np.zeros_like(self.w3, dtype=np.float32); self.vW3 = np.zeros_like(self.w3, dtype=np.float32)
            self.mB1 = np.zeros_like(self.b1, dtype=np.float32); self.vB1 = np.zeros_like(self.b1, dtype=np.float32)
            self.mB2 = np.zeros_like(self.b2, dtype=np.float32); self.vB2 = np.zeros_like(self.b2, dtype=np.float32)
            self.mB3 = np.zeros_like(self.b3, dtype=np.float32); self.vB3 = np.zeros_like(self.b3, dtype=np.float32)
            self.t_adam = 0

        self.samples = np.zeros((NB_SAMPLES_MAX, self.TAILLE_SAMPLE), dtype=np.float32)
        self.samples_count = 0  # Nombre réel de samples stockés
        self.head = 0  # Index circulaire (tête)
        self.W1_target, self.W2_target, self.W3_target = self.w1.copy(), self.w2.copy(), self.w3.copy()
        self.B1_target, self.B2_target, self.B3_target = self.b1.copy(), self.b2.copy(), self.b3.copy()
        self.ct_majs_reseau = 0

    def calcul_couche_sortie(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
            A0 = np.array(state)
            Z1 = np.dot(self.w1, A0) + self.b1
            A1 = relu(Z1)
            Z2 = np.dot(self.w2, A1) + self.b2
            A2 = relu(Z2)
            Z3 = np.dot(self.w3, A2) + self.b3
            A3 = Z3
            return A3

    def ajout_sample(self, sample: NDArray[np.float32]):
        self.samples[self.head] = sample
        self.head = (self.head + 1) % len(self.samples)
        if self.samples_count < len(self.samples):
            self.samples_count += 1

    def entrainement_reseau(self, numero_partie: int):
        if self.samples_count >= NB_SAMPLES_DEBUT_ENTRAINEMENT:
            for _ in range(NB_ENTRAINEMENT_BATCH):
                indices = np.random.choice(self.samples_count, TAILLE_BATCHS, replace=False)
                selection = self.samples[indices]

                # --- 1. Extraction des données du batch (vectorisé) ---
                states1_batch = selection[:, :self.TAILLE_STATE]
                states2_batch = selection[:, self.TAILLE_STATE:2*self.TAILLE_STATE]
                actions_batch = selection[:, 2*self.TAILLE_STATE].astype(int)
                rewards_batch = selection[:, 2*self.TAILLE_STATE+1]
                terminal_states_batch = selection[:, 2*self.TAILLE_STATE+2].astype(bool)

                # --- 2. Forward Pass pour Q_target (réseau cible) ---
                Q_target = np.zeros(TAILLE_BATCHS, dtype=np.float32)
                non_terminal_mask = ~terminal_states_batch
                Z1_s2 = states2_batch @ self.w1.T + self.b1
                A1_s2 = relu(Z1_s2)
                Z2_s2 = A1_s2 @ self.w2.T + self.b2
                A2_s2 = relu(Z2_s2)
                Z3_s2 = A2_s2 @ self.w3.T + self.b3
                if np.any(non_terminal_mask):
                    Z1_target = states2_batch @ self.W1_target.T + self.B1_target
                    A1_target = relu(Z1_target)
                    Z2_target = A1_target @ self.W2_target.T + self.B2_target
                    A2_target = relu(Z2_target)
                    Z3_target = A2_target @ self.W3_target.T + self.B3_target
                    best_actions = np.argmax(Z3_s2, axis=1)
                    Q_target[non_terminal_mask] = (rewards_batch[non_terminal_mask]
                        + gamma * Z3_target[non_terminal_mask, best_actions[non_terminal_mask]])
                Q_target[terminal_states_batch] = rewards_batch[terminal_states_batch]

                # --- 3. Forward Pass pour Q (réseau principal) ---
                Z1_s1 = states1_batch @ self.w1.T + self.b1
                A1_s1 = relu(Z1_s1)
                Z2_s1 = A1_s1 @ self.w2.T + self.b2
                A2_s1 = relu(Z2_s1)
                Z3_s1 = A2_s1 @ self.w3.T + self.b3
                Q = Z3_s1[np.arange(TAILLE_BATCHS), actions_batch - 1]  # Extraction des Q pour chaque action

                # --- 4. Backward Pass (vectorisé) ---
                erreur = Q - Q_target

                gradientaC = np.where(
                    np.abs(erreur) <= DELTA_HUBER_LOSS,
                    erreur,
                    DELTA_HUBER_LOSS * np.sign(erreur)
                )

                delta3 = np.zeros((TAILLE_BATCHS, self.NB_ACTIONS_POSSIBLES), dtype=np.float32)
                delta3[np.arange(TAILLE_BATCHS), actions_batch - 1] = gradientaC

                delta2 = (delta3 @ self.w3) * relu_derivative(Z2_s1)
                delta1 = (delta2 @ self.w2) * relu_derivative(Z1_s1)

                # Gradients pour W1, W2, B1, B2
                dW1 = (delta1.T @ states1_batch / TAILLE_BATCHS).astype(np.float32, copy=False)
                dW2 = (delta2.T @ A1_s1 / TAILLE_BATCHS).astype(np.float32, copy=False)
                dW3 = (delta3.T @ A2_s1 / TAILLE_BATCHS).astype(np.float32, copy=False)
                dB1 = np.sum(delta1, axis=0).astype(np.float32, copy=False) / TAILLE_BATCHS
                dB2 = np.sum(delta2, axis=0).astype(np.float32, copy=False) / TAILLE_BATCHS
                dB3 = np.sum(delta3, axis=0).astype(np.float32, copy=False) / TAILLE_BATCHS
                norme = np.sqrt(
                    np.sum(dW1**2) +
                    np.sum(dW2**2) +
                    np.sum(dW3**2) +
                    np.sum(dB1**2) +
                    np.sum(dB2**2) +
                    np.sum(dB3**2)
                )
                if norme > MAX_NORME_GRADIENT:
                    facteur = MAX_NORME_GRADIENT / norme
                    dW1 *= facteur
                    dW2 *= facteur
                    dW3 *= facteur
                    dB1 *= facteur
                    dB2 *= facteur
                    dB3 *= facteur

                if self.t_adam % 1000 == 0:
                    print()
                    print(
                    f"Q [{np.min(Q):.3f}, {np.mean(Q):.3f}, {np.max(Q):.3f}] | "
                    f"Target [{np.min(Q_target):.3f}, {np.mean(Q_target):.3f}, {np.max(Q_target):.3f}] | "
                    f"Erreur {np.mean(np.abs(Q - Q_target)):.3f} | "
                    f"Grad {norme:.3f}")
                    print(
                    f"Reward [{np.min(rewards_batch):.3f}, "
                    f"{np.mean(rewards_batch):.3f}, "
                    f"{np.max(rewards_batch):.3f}]"
                    )
                    print()

                # --- 5. Mise à jour des poids ---
                self.t_adam += 1
                adam_update(self.w1, dW1, self.mW1, self.vW1, self.t_adam, learning_rate)
                adam_update(self.w2, dW2, self.mW2, self.vW2, self.t_adam, learning_rate)
                adam_update(self.w3, dW3, self.mW3, self.vW3, self.t_adam, learning_rate)
                adam_update(self.b1, dB1, self.mB1, self.vB1, self.t_adam, learning_rate)
                adam_update(self.b2, dB2, self.mB2, self.vB2, self.t_adam, learning_rate)
                adam_update(self.b3, dB3, self.mB3, self.vB3, self.t_adam, learning_rate)

                # Mise à jour du réseau cible
                self.ct_majs_reseau += 1
                if self.ct_majs_reseau % ACTU_W_TARGET == 0:
                    self.W1_target, self.W2_target, self.W3_target = self.w1.copy(), self.w2.copy(), self.w3.copy()
                    self.B1_target, self.B2_target, self.B3_target = self.b1.copy(), self.b2.copy(), self.b3.copy()


                if self.ct_majs_reseau % PERIODE_STOCKAGE_PC == 0:
                    np.savez(self.nom_fichier, W1=self.w1, W2=self.w2, W3=self.w3, B1=self.b1, B2=self.b2, B3=self.b3, mW1=self.mW1, mW2=self.mW2, mW3=self.mW3, mB1=self.mB1, mB2=self.mB2, mB3=self.mB3, vW1=self.vW1, vW2=self.vW2, vW3=self.vW3, vB1=self.vB1, vB2=self.vB2, vB3=self.vB3, t_adam=self.t_adam)
                    print(f"partie : {numero_partie}   Poids, biais exportés dans {self.nom_fichier}")