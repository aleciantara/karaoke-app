from sqlalchemy import create_engine, Column, String, JSON, Integer
from sqlalchemy.orm import declarative_base, Session

engine = create_engine("sqlite:///karaoke.db")
Base = declarative_base()

class Song(Base):
    __tablename__ = "songs"
    id          = Column(String, primary_key=True)
    title       = Column(String)
    youtube_url = Column(String)
    thumbnail   = Column(String)
    words       = Column(JSON)   # [{word, start, end}]
    lyrics_raw  = Column(String)

class QueueItem(Base):
    __tablename__ = "queue"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    song_id   = Column(String)
    singer    = Column(String)
    position  = Column(Integer)

Base.metadata.create_all(engine)