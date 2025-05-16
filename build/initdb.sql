CREATE TABLE IF NOT EXISTS url_analytics (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url TEXT NOT NULL, 
    aggressive_percent INTEGER NOT NULL,
    aggitation_percent INTEGER NOT NULL,
    mat_percent INTEGER NOT NULL,
    bias_percent INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS block_replacements (
    id SERIAL PRIMARY KEY,
    site TEXT NOT NULL,
    aggressive_word_count INT NOT NULL DEFAULT 0,
    block_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(site, block_hash)
);

CREATE TABLE IF NOT EXISTS site_ratings (
    id SERIAL PRIMARY KEY,
    site TEXT NOT NULL UNIQUE,
    aggressive_word_count INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_site_rating()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO site_ratings (site, aggressive_word_count)
    VALUES (NEW.site, NEW.aggressive_word_count)
    ON CONFLICT (site) DO UPDATE SET
        aggressive_word_count = site_ratings.aggressive_word_count + EXCLUDED.aggressive_word_count,
        updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_site_rating
AFTER INSERT ON block_replacements
FOR EACH ROW
EXECUTE FUNCTION update_site_rating();