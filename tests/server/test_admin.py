"""
Admins see every account, with what it uses, and can disable one (it is
signed out at once) or enable it again.
"""

from helpers import PASSWORD, make_admin, signup


def test_admins_list_accounts(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    for name in ("ada", "adam", "bob"):
        signup(new_browser(), username=name, email=f"{name}@example.org")
    users = admin.get("/api/admin/users").json()
    assert [u["username"] for u in users][:3] == ["bob", "adam", "ada"]
    assert users[0]["storage_bytes_used"] == 0
    matched = [u["username"] for u in admin.get("/api/admin/users?q=AD").json()]
    assert matched == ["adam", "ada", "admin"]
    assert [u["username"] for u in admin.get("/api/admin/users?q=bob@").json()] == [
        "bob"
    ]
    # Wildcards are plain characters.
    assert admin.get("/api/admin/users?q=%25").json() == []


def test_only_admins_see_accounts(new_browser):
    ada = new_browser()
    signup(ada)
    assert ada.get("/api/admin/users").status_code == 403


def test_disabling_signs_people_out(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    ada = new_browser()
    signup(ada)
    assert ada.get("/api/projects").status_code == 200
    ada_id = ada.get("/api/auth/session").json()["user"]["id"]

    assert (
        admin.put(
            f"/api/admin/users/{ada_id}/status", json={"status": "disabled"}
        ).status_code
        == 204
    )
    assert ada.get("/api/projects").status_code == 401
    again = new_browser()
    login = {"username": "ada", "password": PASSWORD}
    assert again.post("/api/auth/login", json=login).status_code == 401

    assert (
        admin.put(
            f"/api/admin/users/{ada_id}/status", json={"status": "active"}
        ).status_code
        == 204
    )
    assert again.post("/api/auth/login", json=login).status_code == 200

    admin_id = admin.get("/api/auth/session").json()["user"]["id"]
    selfie = admin.put(
        f"/api/admin/users/{admin_id}/status", json={"status": "disabled"}
    )
    assert selfie.status_code == 409
