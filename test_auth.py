import pytest
import os
from Code.auth.auth_service import (
    hash_password,
    verify_password,
    register_user,
    authenticate_user,
    load_users,
    save_users,
    get_user_role,
    promote_to_admin,
)


def setup_function():
    if os.path.exists("users.json"):
        os.remove("users.json")


def test_hash_password():
    password = "password123"
    hashed = hash_password(password)

    assert hashed != password, "Password should not be stored in plaintext"
    assert isinstance(hashed, str), "Hashed password should be a string"


def test_verify_password_correct():
    password = "password123"
    hashed = hash_password(password)

    assert verify_password(
        password, hashed) is True, "Correct password should validate"


def test_verify_password_wrong():
    password = "password123"
    hashed = hash_password(password)

    assert verify_password(
        "wrongpass", hashed) is False, "Wrong password should fail"


def test_register_user_rejects_short_password():
    with pytest.raises(ValueError):
        register_user("diogo", "1234567")


def test_register_user_accepts_valid_password():
    result = register_user("diogo", "password123")

    assert result is not None, "Valid user should be created"


def test_register_user_duplicate():
    register_user("diogo", "password123")

    with pytest.raises(ValueError):
        register_user("diogo", "password123")


def test_register_user_persists_user():
    register_user("diogo", "password123")
    users = load_users()

    assert "diogo" in users, "User should be saved in users.json"


def test_register_user_stores_hashed_password():
    register_user("diogo", "password123")
    users = load_users()

    assert users["diogo"]["password"] != "password123", "Stored password should be hashed"


def test_register_user_default_role_is_user():
    register_user("diogo", "password123")
    users = load_users()

    assert users["diogo"]["role"] == "user", "New users should be registered as regular users"


def test_register_user_accepts_password_with_8_chars():
    result = register_user("ana", "12345678")

    assert result == "ana", "Password with exactly 8 characters should be accepted"


def test_authenticate_user_valid():
    register_user("maria", "password123")
    result = authenticate_user("maria", "password123")

    assert result is True, "Valid credentials should authenticate"


def test_authenticate_user_invalid_password():
    register_user("joao", "password123")
    result = authenticate_user("joao", "wrongpass")

    assert result is False, "Wrong password should not authenticate"


def test_authenticate_unknown_user():
    result = authenticate_user("unknown_user", "password123")

    assert result is False, "Unknown user should not authenticate"


def test_get_user_role_returns_user():
    register_user("diogo", "password123")

    assert get_user_role("diogo") == "user"


def test_get_user_role_returns_none_for_unknown_user():
    assert get_user_role("unknown_user") is None


def test_admin_can_promote_one_specific_user():
    users = {
        "admin": {
            "password": hash_password("admin123"),
            "role": "admin"
        }
    }
    save_users(users)

    register_user("diogo", "password123")
    register_user("maria", "password123")

    result = promote_to_admin("admin", "diogo")

    assert result is True
    assert get_user_role("diogo") == "admin"
    assert get_user_role("maria") == "user"


def test_promote_to_admin_returns_false_if_target_is_already_admin():
    users = {
        "admin": {
            "password": hash_password("admin123"),
            "role": "admin"
        },
        "diogo": {
            "password": hash_password("password123"),
            "role": "admin"
        }
    }
    save_users(users)

    result = promote_to_admin("admin", "diogo")

    assert result is False


def test_non_admin_cannot_promote_user():
    register_user("user1", "password123")
    register_user("user2", "password123")

    with pytest.raises(PermissionError):
        promote_to_admin("user1", "user2")


def test_promote_unknown_user_raises_error():
    users = {
        "admin": {
            "password": hash_password("admin123"),
            "role": "admin"
        }
    }
    save_users(users)

    with pytest.raises(ValueError):
        promote_to_admin("admin", "ghost")
