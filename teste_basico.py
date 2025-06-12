import unittest
import bcrypt #bcrypt: usado para testar senha criptografada.
import pyotp #pyotp: para gerar tokens temporários (TOTP) no MFA.
from unittest.mock import patch, mock_open #patch e mock_open: simulam funções (ex: leitura/escrita de arquivos).
from app import load_users, save_users, register, autenticar, verificar_mfa


class TestUserFunctions(unittest.TestCase):
    # Usa o decorador @patch para simular (mockar) a função 'open' embutida do Python.
    # Isso permite testar a função `load_users()` sem precisar acessar um arquivo real.
    # `new_callable=mock_open` cria um objeto simulado de arquivo.
    # `read_data='...'` define o conteúdo que será retornado quando o arquivo for "lido".
    @patch("builtins.open", new_callable=mock_open, read_data='{"user@example.com": {"password": "abc", "mfa_secret": "xyz", "verified": false}}')
    def test_load_users(self, mock_file):
        users = load_users()
        self.assertIn("user@example.com", users) # Verifica se o e-mail "user@example.com" está presente nos dados retornados.
        self.assertFalse(users["user@example.com"]["verified"])   # Verifica se o campo "verified" para esse usuário está definido como False (não verificado).


    #Simula salvar um usuário no users.json. Verifica se o método .write() foi chamado, ou seja, se a escrita realmente ocorreu.
    @patch("builtins.open", new_callable=mock_open)
    def test_save_users(self, mock_file):
        users = {"newuser@example.com": {"password": "hash", "mfa_secret": "secret", "verified": True}}
        save_users(users)
        mock_file().write.assert_called()  # Verifica que a escrita ocorreu


    #Verifica se a mensagem de sucesso aparece e se o QR foi retornado corretamente.
    @patch("app.load_users", return_value={})
    @patch("app.save_users")
    @patch("app.create_qr_code", return_value="QR_CODE_IMG")
    def test_register_success(self, mock_qr, mock_save, mock_load):
        email = "test@example.com"
        password = "123456"
        msg, qr = register(email, password)
        self.assertIn("Usuário criado", msg)
        self.assertEqual(qr, "QR_CODE_IMG")

# Simula um login correto com senha válida (criptografada). Verifica se:A mensagem pede o token MFA.O e-mail é retornado corretamente.O login não é concluído ainda (aguarda MFA).
    @patch("app.load_users")
    def test_autenticar_success(self, mock_load):
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

    @patch("app.save_users")
    @patch("app.load_users")
    # Gera um mfa_secret real, cria um token válido com pyotp, e testa se a verificação: Dá certo com esse token, Retorna a mensagem correta.
    def test_verificar_mfa_success(self, mock_load, mock_save):
        mfa_secret = pyotp.random_base32()
        totp = pyotp.TOTP(mfa_secret)
        token = totp.now()
        mock_load.return_value = {"user@example.com": {"password": "x", "mfa_secret": mfa_secret, "verified": False}}

        msg, success = verificar_mfa("user@example.com", token)
        self.assertTrue(success)
        self.assertIn("MFA verificado", msg)

if __name__ == '__main__':
    unittest.main()
