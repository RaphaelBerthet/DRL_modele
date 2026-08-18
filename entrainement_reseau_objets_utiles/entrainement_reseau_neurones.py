import numpy as np
from .adam_update import adam_update
from .relu import relu, relu_derivative
from .parametres import TAILLE_STATE, NB_ACTIONS_POSSIBLES, NB_SAMPLES_MAX, NB_SAMPLES_DEBUT_ENTRAINEMENT, TAILLE_BATCHS, NB_ENTRAINEMENT_BATCH, gamma, learning_rate, ACTU_W_TARGET, PERIODE_STOCKAGE_PC, NB_NEURONES_LAYER1, NB_NEURONES_LAYER2, MAX_NORME_GRADIENT, DELTA_HUBER_LOSS


TAILLE_SAMPLE = 2 * TAILLE_STATE + 3


class Reseau_neurones:
    def __init__(self):
        try:
            data = np.load('reseau_neurones.npz')
            self.W1 = data['W1']
            self.W2 = data['W2']
            self.W3 = data['W3']
            self.B1 = data['B1']
            self.B2 = data['B2']
            self.B3 = data['B3']
            self.mW1 = data['mW1']; self.vW1 = data['vW1']
            self.mW2 = data['mW2']; self.vW2 = data['vW2']
            self.mW3 = data['mW3']; self.vW3 = data['vW3']
            self.mB1 = data['mB1']; self.vB1 = data['vB1']
            self.mB2 = data['mB2']; self.vB2 = data['vB2']
            self.mB3 = data['mB3']; self.vB3 = data['vB3']
            self.t_adam = int(data['t_adam'])

        except FileNotFoundError:
            self.W1 = np.random.randn(NB_NEURONES_LAYER1, TAILLE_STATE) * np.sqrt(2 / TAILLE_STATE)  # He init pour ReLU
            self.W2 = np.random.randn(NB_NEURONES_LAYER2, NB_NEURONES_LAYER1) * np.sqrt(2 / NB_NEURONES_LAYER1)
            self.W3 = np.random.randn(NB_ACTIONS_POSSIBLES, NB_NEURONES_LAYER2) * np.sqrt(2 / NB_NEURONES_LAYER2)
            self.B1 = np.zeros(NB_NEURONES_LAYER1)
            self.B2 = np.zeros(NB_NEURONES_LAYER2)
            self.B3 = np.zeros(NB_ACTIONS_POSSIBLES)
            self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
            self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
            self.mW3 = np.zeros_like(self.W3); self.vW3 = np.zeros_like(self.W3)
            self.mB1 = np.zeros_like(self.B1); self.vB1 = np.zeros_like(self.B1)
            self.mB2 = np.zeros_like(self.B2); self.vB2 = np.zeros_like(self.B2)
            self.mB3 = np.zeros_like(self.B3); self.vB3 = np.zeros_like(self.B3)
            self.t_adam = 0

        self.samples = np.zeros((NB_SAMPLES_MAX, TAILLE_SAMPLE), dtype=np.float32)  # 300+300+3
        self.samples_count = 0  # Nombre réel de samples stockés
        self.head = 0  # Index circulaire (tête)
        self.W1_target, self.W2_target, self.W3_target = self.W1.copy(), self.W2.copy(), self.W3.copy()
        self.B1_target, self.B2_target, self.B3_target = self.B1.copy(), self.B2.copy(), self.B3.copy()
        self.ct_majs_reseau = 0

    def calcul_couche_sortie(self, state):
            A0 = np.array(state)
            Z1 = np.dot(self.W1, A0) + self.B1
            A1 = relu(Z1)
            Z2 = np.dot(self.W2, A1) + self.B2
            A2 = relu(Z2)
            Z3 = np.dot(self.W3, A2) + self.B3
            A3 = Z3
            return A3

    def ajout_sample(self, sample):
        self.samples[self.head] = sample
        self.head = (self.head + 1) % len(self.samples)
        if self.samples_count < len(self.samples):
            self.samples_count += 1

    def entrainement_reseau(self, numero_partie):
        if self.samples_count >= NB_SAMPLES_DEBUT_ENTRAINEMENT:
            for _ in range(NB_ENTRAINEMENT_BATCH):
                indices = np.random.choice(self.samples_count, TAILLE_BATCHS, replace=False)
                selection = self.samples[indices]

                # --- 1. Extraction des données du batch (vectorisé) ---
                states1_batch = selection[:, :TAILLE_STATE]
                states2_batch = selection[:, TAILLE_STATE:2*TAILLE_STATE]
                actions_batch = selection[:, 2*TAILLE_STATE].astype(int)
                rewards_batch = selection[:, 2*TAILLE_STATE+1]
                terminal_states_batch = selection[:, 2*TAILLE_STATE+2].astype(bool)

                # --- 2. Forward Pass pour Q_target (réseau cible) ---
                Q_target = np.zeros(TAILLE_BATCHS)
                non_terminal_mask = ~terminal_states_batch
                Z1_s2 = states2_batch @ self.W1.T + self.B1
                A1_s2 = relu(Z1_s2)
                Z2_s2 = A1_s2 @ self.W2.T + self.B2
                A2_s2 = relu(Z2_s2)
                Z3_s2 = A2_s2 @ self.W3.T + self.B3
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
                Z1_s1 = states1_batch @ self.W1.T + self.B1
                A1_s1 = relu(Z1_s1)
                Z2_s1 = A1_s1 @ self.W2.T + self.B2
                A2_s1 = relu(Z2_s1)
                Z3_s1 = A2_s1 @ self.W3.T + self.B3
                Q = Z3_s1[np.arange(TAILLE_BATCHS), actions_batch - 1]  # Extraction des Q pour chaque action

                # --- 4. Backward Pass (vectorisé) ---
                erreur = Q - Q_target

                gradientaC = np.where(
                    np.abs(erreur) <= DELTA_HUBER_LOSS,
                    erreur,
                    DELTA_HUBER_LOSS * np.sign(erreur)
                )

                delta3 = np.zeros((TAILLE_BATCHS, NB_ACTIONS_POSSIBLES))
                delta3[np.arange(TAILLE_BATCHS), actions_batch - 1] = gradientaC

                delta2 = (delta3 @ self.W3) * relu_derivative(Z2_s1)
                delta1 = (delta2 @ self.W2) * relu_derivative(Z1_s1)

                # Gradients pour W1, W2, B1, B2
                dW1 = delta1.T @ states1_batch
                dW2 = delta2.T @ A1_s1
                dW3 = delta3.T @ A2_s1
                dB1 = np.sum(delta1, axis=0)
                dB2 = np.sum(delta2, axis=0)
                dB3 = np.sum(delta3, axis=0)
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
                
                # --- 5. Mise à jour des poids ---
                self.t_adam += 1
                adam_update(self.W1, dW1 / TAILLE_BATCHS, self.mW1, self.vW1, self.t_adam, learning_rate)
                adam_update(self.W2, dW2 / TAILLE_BATCHS, self.mW2, self.vW2, self.t_adam, learning_rate)
                adam_update(self.W3, dW3 / TAILLE_BATCHS, self.mW3, self.vW3, self.t_adam, learning_rate)
                adam_update(self.B1, dB1 / TAILLE_BATCHS, self.mB1, self.vB1, self.t_adam, learning_rate)
                adam_update(self.B2, dB2 / TAILLE_BATCHS, self.mB2, self.vB2, self.t_adam, learning_rate)
                adam_update(self.B3, dB3 / TAILLE_BATCHS, self.mB3, self.vB3, self.t_adam, learning_rate)

                # Mise à jour du réseau cible
                self.ct_majs_reseau += 1
                if self.ct_majs_reseau % ACTU_W_TARGET == 0:
                    self.W1_target, self.W2_target, self.W3_target = self.W1.copy(), self.W2.copy(), self.W3.copy()
                    self.B1_target, self.B2_target, self.B3_target = self.B1.copy(), self.B2.copy(), self.B3.copy()


                if self.ct_majs_reseau % PERIODE_STOCKAGE_PC == 0:
                    np.savez("reseau_neurones.npz", W1=self.W1, W2=self.W2, W3=self.W3, B1=self.B1, B2=self.B2, B3=self.B3, mW1=self.mW1, mW2=self.mW2, mW3=self.mW3, mB1=self.mB1, mB2=self.mB2, mB3=self.mB3, vW1=self.vW1, vW2=self.vW2, vW3=self.vW3, vB1=self.vB1, vB2=self.vB2, vB3=self.vB3, t_adam=self.t_adam)
                    print(f"partie : {numero_partie}   Poids, biais exportés dans reseau_neurones.npz")