CREATE TABLE IF NOT EXISTS url_analytics (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url TEXT NOT NULL, 
    aggressive_percent INTEGER NOT NULL,
    aggitation_percent INTEGER NOT NULL,
    mat_percent INTEGER NOT NULL,
    bias_percent INTEGER NOT NULL
);