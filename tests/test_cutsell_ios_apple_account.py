"""iPhone: Sign in with Apple is on (no hidden flag), logout exists, an Apple session that ends
shows the welcome screen instead of a new empty account, and the entitlement is declared."""
from pathlib import Path

IOS = Path(__file__).resolve().parents[1] / "mobile" / "ios"
APP = IOS / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_apple_sign_in_is_not_hidden_behind_a_flag():
    state = read("AppState.swift")
    assert "cutsell.auth.apple.enabled" not in state
    assert "persistentAppleAuthEnabled" not in state
    assert "func signInWithApple(identityToken: String, nonce: String?)" in state


def test_button_uses_hashed_nonce_and_apple_button():
    button = read("AppleSignIn.swift")
    assert "SignInWithAppleButton(.signIn)" in button
    assert "request.nonce = hashedNonce" in button
    assert "AppleNonce.sha256(AppleNonce.random())" in button
    assert "appState.signInWithApple(identityToken: identityToken, nonce: nonce)" in button


def test_guest_session_goes_with_the_apple_request():
    # APIClient.request adds the current bearer token, so the server can adopt the guest account.
    client = read("APIClient.swift")
    assert 'request.setValue("Bearer \\(token)", forHTTPHeaderField: "Authorization")' in client
    assert '"/v1/auth/apple"' in client and 'accountProvider: "apple"' in client


def test_logout_and_account_actions_live_in_you():
    assert '"/v1/auth/logout"' in read("APIClient.swift")
    you = read("YouView.swift")
    assert 'SecondaryButton(title: "Log Out"' in you and 'PrimaryButton(title: "Sign in with Apple"' in you
    assert 'Button("Delete Account", role: .destructive)' in you
    assert "appState.session?.isAppleAccount == true" in you
    assert "AppleSignInSheet()" in you and "await appState.logout()" in you


def test_ended_apple_session_shows_welcome_not_a_new_account():
    state = read("AppState.swift")
    branch = state.split("if saved.isAppleAccount {", 1)[1].split("}", 1)[0]
    assert "needsSignIn = true" in branch and "return" in branch
    root = read("CutSellApp.swift")
    assert "appState.needsSignIn" in root and "WelcomeView()" in root


def test_entitlement_declared():
    assert "com.apple.developer.applesignin" in (IOS / "Config" / "CutSell.entitlements").read_text()
    assert "CODE_SIGN_ENTITLEMENTS: Config/CutSell.entitlements" in (IOS / "project.yml").read_text()


def test_saved_sessions_without_provider_still_decode():
    models = read("Models.swift")
    assert "var accountProvider: String? = nil" in models
