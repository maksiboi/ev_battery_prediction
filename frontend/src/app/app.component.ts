import { Component } from '@angular/core';
import {FormBuilder, FormGroup, ReactiveFormsModule, Validators} from '@angular/forms';
import { PredictionService } from './prediction.service';
import {NgForOf, NgIf, NgSwitch, NgSwitchCase} from '@angular/common';
import {Prediction} from './prediction';

interface VehicleOption {
  id: string;
  description: string;
}



@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  imports: [
    ReactiveFormsModule,
    NgSwitchCase,
    NgIf,
    NgSwitch,
    NgForOf
  ],
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  activeTab = 'range'
  rangeForm: FormGroup
  consumptionForm: FormGroup
  batteryRangeForm: FormGroup
  prediction: any = null
  loading = false
  error: string | null = null
  vehicles: VehicleOption[] = [
    { id: 'EV0', description: 'BMW i3, defensive driving style' },
    { id: 'EV1', description: 'BMW i3, normal driving style' },
    { id: 'EV2', description: 'BMW i3, aggressive driving style' },
    { id: 'EV3', description: 'VW ID3, defensive driving style' },
    { id: 'EV4', description: 'VW ID3, normal driving style' },
    { id: 'EV5', description: 'VW ID3, aggressive driving style' },
    { id: 'EV6', description: 'VW ID4, defensive driving style' },
    { id: 'EV7', description: 'VW ID4, normal driving style' },
    { id: 'EV8', description: 'VW ID4, aggressive driving style' }
  ]

  constructor(
    private fb: FormBuilder,
    private predictionService: PredictionService
  ) {
    this.rangeForm = this.fb.group({
      vehicle_id: ['', Validators.required],
      trip_plan: ['amurrio-durango', Validators.required],
      simulation_step: [494.0, Validators.required],
      acceleration: [0, Validators.required],
      actual_battery_capacity_wh: [0, Validators.required],
      state_of_charge: [0, Validators.required],
      speed: [0, Validators.required],
      total_energy_consumed_wh: [0, Validators.required],
      total_energy_regenerated_wh: [0, Validators.required],
      completed_distance: [0, Validators.required],
      traffic_factor: [0, Validators.required],
      wind: [0, Validators.required],
      remaining_range: [0, Validators.required]
    });

    this.consumptionForm = this.fb.group({
      speed: [0, Validators.required],
      acceleration: [0, Validators.required],
      road_slope: [0, Validators.required],
      auxiliaries: [0, Validators.required],
      traffic_factor: [0, Validators.required],
      wind: [0, Validators.required],
      total_energy_consumed_wh: [0, Validators.required]
    });

    this.batteryRangeForm = this.fb.group({
      speed: [0, Validators.required],
      acceleration: [0, Validators.required],
      completed_distance: [0, Validators.required],
      alt: [0, Validators.required],
      road_slope: [0, Validators.required],
      wind: [0, Validators.required],
      traffic_factor: [0, Validators.required],
      occupancy: [0, Validators.required],
      auxiliaries: [0, Validators.required],
      remaining_range: [0, Validators.required]
    });
  }

  setActiveTab(tab: string) {
    this.activeTab = tab;
    this.prediction = null;
    this.error = null;
  }

  onSubmit() {
    this.loading = true;
    this.prediction = null;
    this.error = null;

    let observable;
    let formData;

    switch (this.activeTab) {
      case 'range':
        formData = this.rangeForm.value;
        observable = this.predictionService.predictRange(formData);
        break;
      case 'consumption':
        formData = this.consumptionForm.value;
        observable = this.predictionService.predictConsumption(formData);
        break;
      case 'batteryRange':
        formData = this.batteryRangeForm.value;
        observable = this.predictionService.predictBatteryRange(formData);
        break;
      default:
        return;
    }

    observable.subscribe({
      next: (response: Prediction) => {
        this.prediction = response.prediction;
        this.loading = false;
      },
      error: (error) => {
        this.error = error.message || error.details || "Unknown error occurred.";
        this.loading = false;
      }
    });
  }
}
