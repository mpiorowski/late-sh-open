# late.sh — open source components

Open source parts of [late.sh](https://late.sh), a cozy terminal clubhouse for developers.

```
ssh late.sh
```

Lofi radio, live chat, arcade games, daily challenges, and a bonsai tree that grows while you code, all in your terminal.

## What's here

| Crate | Description |
|-------|-------------|
| [late-cli](late-cli/) | Companion CLI: streams audio locally with a synced visualizer |

This repo is a read-only mirror of selected crates from the main (private) repository. It is automatically synced on every CLI release.

## Build the CLI from source

```bash
git clone https://github.com/mpiorowski/late-sh-open
cd late-sh-open
cargo build --release --bin late
# binary at target/release/late
```

Or install the prebuilt binary:

```bash
curl -fsSL https://cli.late.sh/install.sh | bash
```

## Privacy

Your SSH key **fingerprint** is your identity, we don't store the full public key. No IP logging, no tracking, no analytics.

Don't trust that? Use a throwaway key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/late_throwaway && ssh -i ~/.ssh/late_throwaway late.sh
```

Zero risk, full experience.

## Links

- [late.sh](https://late.sh)
- [GitHub profile](https://github.com/mpiorowski)
