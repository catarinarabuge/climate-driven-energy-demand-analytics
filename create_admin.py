from Code.auth.auth_service import hash_password, save_users, load_users

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

users = load_users()

users[ADMIN_USERNAME] = {
    "password": hash_password(ADMIN_PASSWORD),
    "role": "admin"
}

save_users(users)

print("Admin criado/atualizado com sucesso.")
print("Username: admin")
print("Password: admin123")
