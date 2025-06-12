import unittest
import bcrypt
import pyotp
from unittest.mock import patch, mock_open
from PIL import Image  # Necessário para simular um objeto de imagem PIL

# Importa funções do módulo 'autenticar'
# Certifique-se de que os nomes das funções estejam corretos no arquivo autenticar.py
from autenticar import load_users, save_users, register, autenticar, verificar_mfa, create_qr_code


class TestUserFunctions(unittest.TestCase):

    # Testa se a função load_users carrega corretamente os dados do JSON simulado
    # O mock de 'open' deve estar no namespace do módulo onde load_users está definido (autenticar.py)
    @patch("autenticar.open", new_callable=mock_open, read_data='{"user@example.com": {"password": "abc", "mfa_secret": "xyz", "verified": false}}')
    def test_load_users(self, mock_file):
        users = load_users()
        self.assertIn("user@example.com", users)
        self.assertFalse(users["user@example.com"]["verified"])

    # Testa se save_users chama o método .write() para salvar os dados
    @patch("autenticar.open", new_callable=mock_open)
    def test_save_users(self, mock_file):
        users = {"newuser@example.com": {"password": "hash", "mfa_secret": "secret", "verified": True}}
        save_users(users)
        mock_file().write.assert_called()

    # Testa o registro de um novo usuário com retorno do QR Code
    # Mocka funções de leitura e escrita de usuários, além da criação do QR
    # O mock para create_qr_code deve estar no módulo 'autenticar', já que a função foi movida para lá
    @patch("autenticar.load_users", return_value={})
    @patch("autenticar.save_users")
    @patch("autenticar.create_qr_code", return_value=Image.new('RGB', (1, 1)))  # Simula objeto PIL.Image
    def test_register_success(self, mock_qr, mock_save, mock_load):
        email = "test@example.com"
        password = "123456"
        msg, qr = register(email, password)
        self.assertIn("Usuário criado", msg)
        self.assertIsInstance(qr, Image.Image)

    # Testa a autenticação com senha correta e usuário existente
    # Espera receber solicitação de MFA
    @patch("autenticar.load_users")
    def test_authenticate_success(self, mock_load):
        hashed_pw = bcrypt.hashpw("senha123".encode(), bcrypt.gensalt()).decode()
        mock_load.return_value = {
            "test@example.com": {
                "password": hashed_pw,
                "mfa_secret": "SECRET",
                "verified": False
            }
        }
        msg, logged, email, success = autenticar("test@example.com", "senha123")
        self.assertEqual(msg, "Digite o token MFA")
        self.assertFalse(logged)
        self.assertEqual(email, "test@example.com")
        self.assertTrue(success)

    # Testa a verificação do código MFA gerado com pyotp
    # O código válido deve marcar o usuário como verificado
    @patch("autenticar.save_users")
    @patch("autenticar.load_users")
    def test_verificar_mfa_success(self, mock_load, mock_save):
        mfa_secret = pyotp.random_base32()
        totp = pyotp.TOTP(mfa_secret)
        token = totp.now()
        mock_load.return_value = {
            "user@example.com": {
                "password": "x",
                "mfa_secret": mfa_secret,
                "verified": False
            }
        }

        msg, success = verificar_mfa("user@example.com", token)
        self.assertTrue(success)
        self.assertIn("MFA verificado", msg)


if __name__ == '__main__':
    unittest.main()
