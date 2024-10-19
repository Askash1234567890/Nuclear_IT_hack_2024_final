from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.analyzer import ExitInterviewAnalyzer

app = FastAPI()
analyzer = ExitInterviewAnalyzer()


class LoadDataRequest(BaseModel):
    data: List[Dict[str, str]]


class AnalyzeRequest(BaseModel):
    words: List[str]


class PersonalStatisticRequest(BaseModel):
    id: int


@app.post("/load_data")
async def load_data(request: LoadDataRequest):
    """Load data into the analyzer.

    :param request: LoadDataRequest object containing the data to be loaded
    :return: Success message
    """
    try:
        df = pd.DataFrame(request.data)
        analyzer.load_data(df)
        return {"message": "Data loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_personal_statistic")
async def get_personal_statistic(request: PersonalStatisticRequest):
    """Get personal statistics for a given ID.

    :param request: PersonalStatisticRequest object containing the ID
    :return: Personal statistics in HTML format
    """
    try:
        html_stat = analyzer.get_personal_statistic(request.id)
        return {"personal_statistic": html_stat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_graphics")
async def get_graphics():
    """Generate graphics for the loaded data. And generate PDF file with the graphics.

    :return: Success message
    """
    try:
        analyzer.get_graphics()
        return {"message": "Graphics generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
