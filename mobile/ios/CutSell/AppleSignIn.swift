import AuthenticationServices
import CryptoKit
import SwiftUI

/// Sign in with Apple button wired to `AppState.signInWithApple`.
///
/// Apple puts the SHA-256 of the nonce we send into the identity token; the server checks
/// that the token carries exactly the nonce value the app sends it, so the hashed value is
/// what goes to both.
struct AppleSignInButton: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.colorScheme) private var colorScheme
    @State private var hashedNonce = ""
    @State private var isWorking = false
    @State private var errorMessage: String?
    var onSignedIn: () -> Void = {}

    var body: some View {
        VStack(spacing: 8) {
            SignInWithAppleButton(.signIn) { request in
                hashedNonce = AppleNonce.sha256(AppleNonce.random())
                request.requestedScopes = [.email]
                request.nonce = hashedNonce
            } onCompletion: { result in
                handle(result)
            }
            .signInWithAppleButtonStyle(colorScheme == .dark ? .white : .black)
            .frame(height: 50)
            .disabled(isWorking)
            .overlay {
                if isWorking { ProgressView() }
            }
            if let errorMessage {
                Text(L10n.tr(errorMessage))
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }
        }
    }

    private func handle(_ result: Result<ASAuthorization, Error>) {
        switch result {
        case .failure(let error):
            if let authError = error as? ASAuthorizationError, authError.code == .canceled { return }
            errorMessage = "Sign in with Apple didn't finish. Please try again."
        case .success(let authorization):
            guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential,
                  let tokenData = credential.identityToken,
                  let identityToken = String(data: tokenData, encoding: .utf8) else {
                errorMessage = "Apple didn't return a sign-in token. Please try again."
                return
            }
            let nonce = hashedNonce
            isWorking = true
            errorMessage = nil
            Task {
                defer { isWorking = false }
                do {
                    try await appState.signInWithApple(identityToken: identityToken, nonce: nonce)
                    onSignedIn()
                } catch {
                    errorMessage = "Couldn't sign in right now. Please try again."
                }
            }
        }
    }
}

enum AppleNonce {
    static func random(length: Int = 32) -> String {
        let charset = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz-._")
        var generator = SystemRandomNumberGenerator()
        return String((0..<length).map { _ in charset[Int.random(in: 0..<charset.count, using: &generator)] })
    }

    static func sha256(_ input: String) -> String {
        SHA256.hash(data: Data(input.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

/// Shown after logging out, after deleting the account, or when an Apple session ended.
struct WelcomeView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        VStack(spacing: 24) {
            Spacer()
            VStack(spacing: 8) {
                Text("CutSell")
                    .font(.largeTitle.bold())
                Text("Sign in to keep your projects on every iPhone you use.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Spacer()
            AppleSignInButton()
            Button("Continue without an account") {
                Task { await appState.continueWithoutAccount() }
            }
            .font(.callout)
        }
        .padding(24)
    }
}

/// Opened from the profile menu while using CutSell without an account.
struct AppleSignInSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Image(systemName: "person.crop.circle.badge.checkmark")
                    .font(.system(size: 48))
                    .foregroundStyle(.tint)
                Text("Keep your projects safe")
                    .font(.title2.bold())
                Text("Sign in with Apple and your projects stay with your account, even if you delete the app or change iPhones. The projects you already made stay with you.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                Spacer()
                AppleSignInButton(onSignedIn: { dismiss() })
            }
            .padding(24)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Not now") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}
