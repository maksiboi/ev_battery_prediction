import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {Prediction} from './prediction';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  private baseUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) {}

  predictRange(data: any): Observable<Prediction> {
    return this.http.post<Prediction>(`${this.baseUrl}/range/remaining`, data);
  }

  predictConsumption(data: any): Observable<Prediction> {
    return this.http.post<Prediction>(`${this.baseUrl}/energy/consumption`, data);
  }

  predictBatteryRange(data: any): Observable<Prediction> {
    return this.http.post<Prediction>(`${this.baseUrl}/battery/range`, data);
  }
}
