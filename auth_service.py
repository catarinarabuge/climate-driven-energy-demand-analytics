import json
import bcrypt
import os

USERS_FILE = "users.json"


def load_users(file_path=USERS_FILE):
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_users(users, file_path=USERS_FILE):
    with open(file_path, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())


def register_user(username, password, file_path=USERS_FILE):
    if not username or not username.strip():
        raise ValueError("Username não pode ser vazio")

    if len(password) < 8:
        raise ValueError("Password deve ter no mínimo 8 caracteres")

    users = load_users(file_path)

    if username in users:
        raise ValueError("User já existente")

    users[username] = {
        "password": hash_password(password),
        "role": "user"
    }

    save_users(users, file_path)
    return username


def authenticate_user(username, password, file_path=USERS_FILE):
    users = load_users(file_path)

    if username not in users:
        return False

    return verify_password(password, users[username]["password"])


def get_user_role(username, file_path=USERS_FILE):
    users = load_users(file_path)

    if username not in users:
        return None

    return users[username]["role"]


def promote_to_admin(current_admin, target_user, file_path=USERS_FILE):
    users = load_users(file_path)

    if current_admin not in users:
        raise ValueError("Admin user does not exist")

    if users[current_admin]["role"] != "admin":
        raise PermissionError("Only admins can promote users")

    if target_user not in users:
        raise ValueError("Target user does not exist")

    if users[target_user]["role"] == "admin":
        return False

    users[target_user]["role"] = "admin"
    save_users(users, file_path)

    return True
