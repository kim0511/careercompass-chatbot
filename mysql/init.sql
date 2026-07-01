CREATE TABLE IF NOT EXISTS chat_history (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  session_id  VARCHAR(64)  NOT NULL,
  question    TEXT         NOT NULL,
  answer      TEXT         NOT NULL,
  status      SMALLINT     NOT NULL DEFAULT 200,
  response_ms INT          DEFAULT NULL,
  created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_session (session_id),
  INDEX idx_status  (status)
);
