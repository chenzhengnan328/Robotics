
#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	  num_particles = 30;  
	  default_random_engine gen;
  	double std_x = std[0];
  	double std_y = std[1];
  	double std_theta = std[2];  
  
  	normal_distribution<double> dist_x(x, std_x);
  	normal_distribution<double> dist_y(y, std_y);
  	normal_distribution<double> dist_theta(theta, std_theta);  
  
  	int i;
   	for(i=0; i<num_particles; i++){
		  Particle temp_particles;
      
		  temp_particles.id = i;
		  temp_particles.x = dist_x(gen);
      temp_particles.y = dist_y(gen);
      temp_particles.theta = dist_theta(gen);
      temp_particles.weight = 1.0;
      	
      particles.push_back(temp_particles);
      weights.push_back(temp_particles.weight);
    }
  	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

	  default_random_engine gen;
  	double std_x = std_pos[0];
  	double std_y = std_pos[1];
  	double std_theta = std_pos[2];  
  

  	double particle_x, particle_y, particle_theta;
  	for(int i=0; i<particles.size(); ++i){
      	double pred_x, pred_y, pred_theta;
      	particle_x = particles[i].x;
      	particle_y = particles[i].y;
      	particle_theta = particles[i].theta;
    	if (fabs(yaw_rate) < 0.0001) {
	    	pred_x = particle_x + velocity * cos(particle_theta) * delta_t;
	    	pred_y = particle_y + velocity * sin(particle_theta) * delta_t;
	    	pred_theta = particle_theta;
	  	} 
      else {
	    	pred_x = particle_x + (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
	   		pred_y = particle_y + (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
	    	pred_theta = particle_theta + (yaw_rate * delta_t);
	  	}
  		normal_distribution<double> dist_x(pred_x, std_x);
  		normal_distribution<double> dist_y(pred_y, std_y);
  		normal_distribution<double> dist_theta(pred_theta, std_theta); 
      
      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);
    }

}

vector<LandmarkObs> ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations, double sensor_range){

  	vector<LandmarkObs> ret;
  	for(int i=0; i<observations.size(); ++i){
    	  double obs_x = observations[i].x;
      	double obs_y = observations[i].y;
      	double min_dist = sensor_range;
      	int index = -1;
      	for(int j=0; j<predicted.size(); j++){
        	  double pred_x = predicted[j].x;
          	double pred_y = predicted[j].y;
          	double distance = dist(obs_x, obs_y, pred_x, pred_y);
          	if(distance < min_dist){
            	index = j;
              min_dist =distance;
            }
        }
      	ret.push_back(predicted[index]);
    }
  	return ret;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  	double weight_normalizer = 0.0;
  	for(int i=0; i<particles.size(); i++){
    	double particle_x = particles[i].x;
      	double particle_y = particles[i].y;
      	double particle_theta = particles[i].theta;
      
      
      	//initial a transformed observation vector
      	vector<LandmarkObs> transformed_obslist;
      	for(int j=0; j<observations.size(); j++){
          	LandmarkObs transformed_obs;
      		  transformed_obs.id = j;
      		  transformed_obs.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
      		  transformed_obs.y = particle_y + (sin(particle_theta) * observations[j].x) + (cos(particle_theta) * observations[j].y);
          	transformed_obslist.push_back(transformed_obs);
        }
      	
      	vector<LandmarkObs> predicted_landmarks;
      	for(int i=0; i<map_landmarks.landmark_list.size(); i++){
        	  auto single_landmark = map_landmarks.landmark_list[i];
          	if(dist(particle_x, particle_y, single_landmark.x_f, single_landmark.y_f) <= sensor_range){
            	predicted_landmarks.push_back(LandmarkObs {single_landmark.id_i, single_landmark.x_f, single_landmark.y_f});
            }
        }
     	vector<LandmarkObs> closest_landmarks = dataAssociation(predicted_landmarks,transformed_obslist, sensor_range);
      double temp_prob = 1.0;
      	
    	double sigma_x = std_landmark[0];
    	double sigma_y = std_landmark[1];
    	double sigma_x_2 = pow(sigma_x, 2);
    	double sigma_y_2 = pow(sigma_y, 2);
    	double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
      	
      for(int k=0; k<transformed_obslist.size(); k++){
      	double trans_obs_x = transformed_obslist[k].x;
      	double trans_obs_y = transformed_obslist[k].y;
      		
        double pred_landmark_x  = closest_landmarks[k].x;
        double pred_landmark_y  = closest_landmarks[k].y;
        double multi_prob = normalizer * exp(-1.0 * ((pow((trans_obs_x - pred_landmark_x), 2)/(2.0 * sigma_x_2)) + (pow((trans_obs_y - pred_landmark_y), 2)/(2.0 * sigma_y_2))));
        temp_prob *= multi_prob; 
        }
      	particles[i].weight = temp_prob;
      	weight_normalizer += temp_prob;
    }
  	
  	for(int i=0; i<particles.size(); i++){
    	particles[i].weight /= weight_normalizer;
      weights[i] = particles[i].weight;
    }

}

void ParticleFilter::resample() {

  vector<Particle> resampled_particles;
	default_random_engine gen;
	uniform_int_distribution<int> particle_index(0, num_particles - 1);
	
  	double max_weight = 0;
  	for(int i=0; i<particles.size(); i++){
    	max_weight = max(max_weight, particles[i].weight);
    }
  	max_weight *= 2;
	int current_index = particle_index(gen);
	double beta = 0.0;
  	
  	for(int j=0; j<particles.size(); j++){
		uniform_real_distribution<double> random_weight(0.0, max_weight);
		beta += random_weight(gen);
      	while(beta > weights[current_index]){
        	beta -= weights[current_index];
          	current_index = (current_index + 1) % num_particles;
        }
      	resampled_particles.push_back(particles[current_index]);
    }
  	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}