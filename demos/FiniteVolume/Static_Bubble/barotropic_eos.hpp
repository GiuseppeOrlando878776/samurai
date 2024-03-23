#ifndef barotropic_eos_hpp
#define barotropic_eos_hpp

/**
 * Implementation of a for linearized EOS
 */
class LinearizedBarotropicEOS {
public:
  LinearizedBarotropicEOS() = default;

  LinearizedBarotropicEOS(const double p0_, const double rho0_, const double c0_); // Constructor which accepts as arguments
                                                                                   // reference pressure, density and speed of sound

  template<typename T>
  T pres_value(const T& rho) const; // Function to actually compute the pressure from the density

  template<typename T>
  T c_value(const T& rho) const; // Function to compute the speed of sound

  template<typename T>
  T rho_value(const T& pres) const; // Function to compute the density from the pressure

  double get_c0() const; // Get the speed of sound

  double get_p0() const; // Get the reference pressure

  double get_rho0() const; // Get the reference density

  void set_c0(const double c0_); // Set the speed of sound

  void set_p0(const double p0_); // Set the reference pressure

  void set_rho0(const double rho0_); // Set the reference density

private:
  double p0;   // Reference pressure
  double rho0; // Reference density
  double c0;   // Speed of sound
};


// Implement the constructor
//
LinearizedBarotropicEOS::LinearizedBarotropicEOS(const double p0_, const double rho0_, const double c0_):
  p0(p0_), rho0(rho0_), c0(c0_) {}


// Implement the pressure value from the density
//
template<typename T>
T LinearizedBarotropicEOS::pres_value(const T& rho) const {
  if(std::isnan(rho)) {
    return nan("");
  }

  return p0 + c0*c0*(rho - rho0);
}


// Implement the speed of sound from the density
//
template<typename T>
T LinearizedBarotropicEOS::c_value(const T& rho) const {
  (void) rho;

  return c0;
}


// Implement the density from the pressure
//
template<typename T>
T LinearizedBarotropicEOS::rho_value(const T& pres) const {
  if(std::isnan(pres)) {
    return nan("");
  }

  return (pres - p0)/(c0*c0) + rho0;
}


// Implement the getter of the speed of sound
//
double LinearizedBarotropicEOS::get_c0() const {
  return c0;
}

// Implement the getter of the reference pressure
//
double LinearizedBarotropicEOS::get_p0() const {
  return p0;
}

// Implement the getter of the reference density
//
double LinearizedBarotropicEOS::get_rho0() const {
  return rho0;
}

// Implement the setter for the speed of sound
//
void LinearizedBarotropicEOS::set_c0(const double c0_) {
  c0 = c0_;
}

// Implement the setter of the reference pressure
//
void LinearizedBarotropicEOS::set_p0(const double p0_) {
  p0 = p0_;
}

// Implement the setter of the reference density
//
void LinearizedBarotropicEOS::set_rho0(const double rho0_) {
  rho0 = rho0_;
}

#endif
