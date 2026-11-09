CREATE TABLE appointments (
    id TEXT PRIMARY KEY,
    tutor_name TEXT NOT NULL,
    pet_name TEXT NOT NULL,
    service TEXT NOT NULL CHECK (service IN ('consulta', 'vacina', 'banho_tosa', 'exame')),
    start_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('scheduled', 'cancelled'))
);

CREATE INDEX appointments_start_at_idx ON appointments (start_at);

CREATE UNIQUE INDEX appointments_active_slot_idx ON appointments (start_at)
    WHERE status = 'scheduled';
