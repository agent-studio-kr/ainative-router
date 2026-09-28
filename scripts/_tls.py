"""사내 TLS 프록시 환경에서 시스템 인증서 저장소를 쓰도록 한다."""
import truststore

truststore.inject_into_ssl()
