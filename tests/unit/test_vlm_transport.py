from src.vlm.transport import create_vlm_session, normalize_proxy_url


def test_normalize_proxy_url_defaults_to_direct():
    assert normalize_proxy_url() == ""
    assert normalize_proxy_url(proxy_port=0) == ""


def test_normalize_proxy_url_accepts_port_and_host_port():
    assert normalize_proxy_url(proxy_port=7890) == "http://127.0.0.1:7890"
    assert normalize_proxy_url(proxy_url="127.0.0.1:7890") == "http://127.0.0.1:7890"
    assert normalize_proxy_url(proxy_url="socks5://127.0.0.1:7891") == "socks5://127.0.0.1:7891"


def test_create_vlm_session_disables_environment_proxy_by_default():
    session = create_vlm_session()

    assert session.trust_env is False
    assert dict(session.proxies) == {}


def test_create_vlm_session_uses_explicit_proxy_only():
    session = create_vlm_session(proxy_port=7890)

    assert session.trust_env is False
    assert session.proxies["http"] == "http://127.0.0.1:7890"
    assert session.proxies["https"] == "http://127.0.0.1:7890"
