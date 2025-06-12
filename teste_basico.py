import unittest
import bcrypt
import pyotp
from unittest.mock import patch, mock_open
from PIL import Image # Necessário para simular um objeto de imagem PIL

# Importar as funções dos módulos 'auth' e 'utils'
# Garanta que as funções 'authenticate' e 'verify_mfa' estejam no seu auth.py
# e que 'create_qr_code' esteja no seu utils.py
from auth import load_users, save_users, register, authenticate, verify_mfa
from utils import create_qr_code # Importar a função real para poder mocká-la corretamente


class TestUserFunctions(unittest.TestCase):

    # Usa o decorador @patch para simular (mockar) a função 'open' embutida do Python.
    # Isso permite testar a função `load_users()` sem precisar acessar um arquivo real.
    # `new_callable=mock_open` cria um objeto simulado de arquivo.
    # `read_data='...'` define o conteúdo que será retornado quando o arquivo for "lido".
    # O mock de 'open' deve estar no módulo 'auth', pois 'load_users' o chama lá
    @patch("auth.open", new_callable=mock_open, read_data='{"user@example.com": {"password": "abc", "mfa_secret": "xyz", "verified": false}}')
    def test_load_users(self, mock_file):
        users = load_users()
        self.assertIn("user@example.com", users)
        self.assertFalse(users["user@example.com"]["verified"])


    # Simula salvar um usuário no users.json. Verifica se o método .write() foi chamado.
    # O mock de 'open' deve estar no módulo 'auth', pois 'save_users' o chama lá
    @patch("auth.open", new_callable=mock_open)
    def test_save_users(self, mock_file):
        users = {"newuser@example.com": {"password": "hash", "mfa_secret": "secret", "verified": True}}
        save_users(users)
        mock_file().write.assert_called()


    # Verifica se a mensagem de sucesso aparece e se o QR foi retornado corretamente.
    # Mocks para load_users e save_users devem estar no módulo 'auth'
    # O mock para create_qr_code deve estar no módulo 'utils'
    @patch("auth.load_users", return_value={}) # auth.register chama auth.load_users
    @patch("auth.save_users") # auth.register chama auth.save_users
    # SIMULA UM OBJETO PIL.Image.Image para o retorno de create_qr_code
    @patch("utils.create_qr_code", return_value=Image.new('RGB', (1, 1)))
    def test_register_success(self, mock_qr, mock_save, mock_load):
        email = "test@example.com"
        password = "123456"
        msg, qr = register(email, password)
        self.assertIn("Usuário criado", msg)
        # O teste agora verifica se o objeto retornado é uma instância de Image.Image
        self.assertIsInstance(qr, Image.Image)


    # Simula um login correto com senha válida (criptografada).
    # O mock para 'load_users' deve estar no módulo 'auth'
    @patch("auth.load_users")
    def test_authenticate_success(self, mock_load):
        hashed_pw = bcrypt.hashpw("senha123".encode(), bcrypt.gensalt()).decode()
        mock_load.return_value = {
            "test@example.com": {
                "password": hashed_pw,
                "mfa_secret": "SECRET",
                "verified": False
            }
        }
        # Chamar 'authenticate' (novo nome)
        msg, logged, email, success = authenticate("test@example.com", "senha123")
        self.assertEqual(msg, "Digite o token MFA")
        self.assertFalse(logged)
        self.assertEqual(email, "test@example.com")
        self.assertTrue(success)


    # Gera um mfa_secret real, cria um token válido com pyotp, e testa a verificação.
    # Mocks para 'save_users' e 'load_users' devem estar no módulo 'auth'
    @patch("auth.save_users")
    @patch("auth.load_users")
    def test_verificar_mfa_success(self, mock_load, mock_save):
        mfa_secret = pyotp.random_base32()
        totp = pyotp.TOTP(mfa_secret)
        token = totp.now()
        mock_load.return_value = {"user@example.com": {"password": "x", "mfa_secret": mfa_secret, "verified": False}}

        # Chamar 'verify_mfa' (função importada diretamente do auth.py)
        msg, success = verify_mfa("user@example.com", token)
        self.assertTrue(success)
        self.assertIn("MFA verificado", msg)

if __name__ == '__main__':
    unittest.main()